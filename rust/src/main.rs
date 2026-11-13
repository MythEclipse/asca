//! CLI ASCA v2: train / eval / sample / inspect.

use asca_rs::bpe::Bpe;
use asca_rs::dense::{DenseCfg, DenseModel};
use asca_rs::model::{Config, Model};
use asca_rs::tensor::Rng;
use asca_rs::train::{self, Data, Split, TrainCfg};

fn arg(args: &[String], name: &str, default: &str) -> String {
    for (i, a) in args.iter().enumerate() {
        if a == name {
            if let Some(v) = args.get(i + 1) {
                return v.clone();
            }
        }
    }
    default.to_string()
}

fn main() {
    asca_rs::prof::init();
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).cloned().unwrap_or_else(|| "help".into());
    match cmd.as_str() {
        "train" => cmd_train(&args),
        "eval" => cmd_eval(&args),
        "sample" => cmd_sample(&args),
        "inspect" => cmd_inspect(&args),
        "bench" => cmd_bench(&args),
        "gpu" => cmd_gpu(&args),
        "decode" => cmd_decode(&args),
        "bpe" => cmd_bpe(&args),
        _ => {
            println!("asca — ASCA v2 (tanpa attention, state konstan, Rust murni)");
            println!("pakai:");
            println!(
                "  asca train   --data FILE [--steps 2000] [--seq-len 128] [--batch 8] [--lr 3e-3]"
            );
            println!("               [--d-model 64] [--n-heads 4] [--n-layers 2] [--n-experts 8]");
            println!("               [--d-ff 128] [--out ckpt.bin] [--eval-every 200]");
            println!("  asca eval    --ckpt FILE --data FILE [--batches 20] [--seq-len 128]");
            println!("  asca sample  --ckpt FILE [--prompt TEXT] [--tokens 200]");
            println!("  asca inspect --ckpt FILE");
            println!(
                "  asca bench   --data FILE [--steps 400] [--seeds 0,1,2] [--batch 4] [--seq-len 128]"
            );
            println!(
                "               [--lr 3e-3] [--d-model 64] [--n-layers 2] [--n-experts 8] [--d-ff 128]"
            );
        }
    }
    if std::env::var("ASCA_PROF").ok().as_deref() == Some("1") {
        println!("\n{}", asca_rs::prof::report());
    }
}

/// Decode autoregresif satu token demi satu, mengukur ms/token pada beberapa panjang
/// konteks. Kalau state benar-benar konstan, biayanya harus datar terhadap konteks —
/// inilah beda mendasar dengan KV cache yang tumbuh O(T).
/// Decode autoregresif satu token demi satu, mengukur ms/token pada beberapa panjang
/// konteks. ASCA membawa state `dh x dh` per head yang konstan; dense membawa KV cache
/// yang tumbuh O(T). Perbandingan ini yang membuat klaim memori punya isi.
fn cmd_decode(args: &[String]) {
    let which = arg(args, "--model", "asca");
    let ctxs: Vec<usize> = arg(args, "--ctx", "128,512,1024,2048")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let steps: usize = arg(args, "--steps", "32").parse().unwrap_or(32);
    let warmup: usize = arg(args, "--warmup", "8").parse().unwrap_or(8);
    let mut rng = Rng::new(7);
    println!("| konteks | ms/token | tok/s | state float |");
    println!("|---|---|---|---|");
    match which.as_str() {
        "dense" => {
            let d_ff: usize = arg(args, "--d-ff", "128").parse().unwrap_or(128);
            let cfg = DenseCfg {
                vocab: vocab_for(args),
                d_model: 64,
                n_heads: 4,
                n_layers: 2,
                d_ff,
            };
            let model = DenseModel::new(cfg, &mut rng);
            for &ctx in ctxs.iter() {
                let mut st = model.new_decode_state();
                for _ in 0..ctx + warmup {
                    let id = rng.below(256) as u32;
                    let _ = model.step(id, &mut st);
                }
                let t0 = std::time::Instant::now();
                for _ in 0..steps {
                    let id = rng.below(256) as u32;
                    let _ = model.step(id, &mut st);
                }
                let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
                let kv = DenseModel::decode_state_floats(&st);
                println!("| {ctx} | {ms:.3} | {:.0} | {kv} |", 1e3 / ms);
            }
        }
        _ => {
            let cfg = build_cfg(args);
            let model = Model::new(cfg, &mut rng);
            let floats: usize = model
                .new_decode_state()
                .delta
                .iter()
                .flatten()
                .map(|v| v.len())
                .sum();
            for &ctx in ctxs.iter() {
                let mut st = model.new_decode_state();
                for _ in 0..ctx + warmup {
                    let id = rng.below(256) as u32;
                    let _ = model.step(id, &mut st);
                }
                let t0 = std::time::Instant::now();
                for _ in 0..steps {
                    let id = rng.below(256) as u32;
                    let _ = model.step(id, &mut st);
                }
                let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
                println!("| {ctx} | {ms:.3} | {:.0} | {floats} |", 1e3 / ms);
            }
        }
    }
}

/// Tokenizer BPE: latih, lalu laporkan kompresi. Tanpa ini model hanya melihat byte,
/// jadi "LLM nyata" belum mungkin.
fn cmd_bpe(args: &[String]) {
    let sub = args.get(2).cloned().unwrap_or_default();
    match sub.as_str() {
        "train" => {
            let path = arg(args, "--data", "");
            let vocab: usize = arg(args, "--vocab", "8192").parse().unwrap();
            let out = arg(args, "--out", "tok.bin");
            let text = std::fs::read(&path).expect("gagal baca data");
            let t0 = std::time::Instant::now();
            let bpe = Bpe::train(&text, vocab);
            let dt = t0.elapsed().as_secs_f64();
            bpe.save(&out).expect("gagal simpan");
            // kompresi pada potongan yang tidak dipakai menghitung: 1 MB terakhir
            let sample = &text[text.len().saturating_sub(1 << 20)..];
            let ids = bpe.encode(sample);
            println!(
                "bpe: {} merge, vocab {} dalam {:.1} s",
                bpe.vocab_size() - 256,
                bpe.vocab_size(),
                dt
            );
            println!(
                "kompresi: {} byte -> {} token ({:.2} byte/token) pada 1 MB terakhir",
                sample.len(),
                ids.len(),
                sample.len() as f64 / ids.len() as f64
            );
            // cek round-trip pada data nyata, bukan hanya data uji
            assert_eq!(bpe.decode(&ids), sample.to_vec(), "round-trip gagal");
            println!("round-trip: OK");
            println!("disimpan: {out}");
        }
        "stats" => {
            let tok = arg(args, "--tokenizer", "tok.bin");
            let path = arg(args, "--data", "");
            let bpe = Bpe::load(&tok).expect("gagal muat tokenizer");
            let text = std::fs::read(&path).expect("gagal baca data");
            let ids = bpe.encode(&text);
            println!(
                "vocab {} | {} byte -> {} token ({:.2} byte/token)",
                bpe.vocab_size(),
                text.len(),
                ids.len(),
                text.len() as f64 / ids.len() as f64
            );
        }
        _ => println!("pakai: asca bpe train --data FILE --vocab 8192 --out tok.bin"),
    }
}

/// Muat data; kalau `--tokenizer` diberikan, teks di-encode jadi subword BPE.
/// Tanpa flag itu, perilakunya tetap seperti sebelumnya (byte sebagai id).
fn load_data(args: &[String]) -> Data {
    let path = arg(args, "--data", "");
    let tok = arg(args, "--tokenizer", "");
    if tok.is_empty() {
        Data::load(&path, 0.05).expect("gagal baca data")
    } else {
        let bpe = Bpe::load(&tok).expect("gagal muat tokenizer");
        Data::load_with_bpe(&path, 0.05, &bpe).expect("gagal baca data")
    }
}

/// Vocab harus sama dengan tokenizer; kalau `--tokenizer` ada, ukurannya yang menang.
fn vocab_for(args: &[String]) -> usize {
    let tok = arg(args, "--tokenizer", "");
    if tok.is_empty() {
        arg(args, "--vocab", "256").parse().unwrap()
    } else {
        Bpe::load(&tok).expect("gagal muat tokenizer").vocab_size()
    }
}

fn build_cfg(args: &[String]) -> Config {
    Config {
        vocab: vocab_for(args),
        d_model: arg(args, "--d-model", "64").parse().unwrap(),
        n_heads: arg(args, "--n-heads", "4").parse().unwrap(),
        n_layers: arg(args, "--n-layers", "2").parse().unwrap(),
        n_experts: arg(args, "--n-experts", "8").parse().unwrap(),
        d_ff: arg(args, "--d-ff", "128").parse().unwrap(),
    }
}

fn cmd_train(args: &[String]) {
    assert!(!arg(args, "--data", "").is_empty(), "butuh --data FILE");
    let out = arg(args, "--out", "asca2.ckpt");
    let cfg = build_cfg(args);
    let tcfg = TrainCfg {
        steps: arg(args, "--steps", "2000").parse().unwrap(),
        seq_len: arg(args, "--seq-len", "128").parse().unwrap(),
        batch: arg(args, "--batch", "8").parse().unwrap(),
        lr: arg(args, "--lr", "3e-3").parse().unwrap(),
        warmup: arg(args, "--warmup", "50").parse().unwrap(),
        eval_every: arg(args, "--eval-every", "200").parse().unwrap(),
        eval_batches: arg(args, "--eval-batches", "8").parse().unwrap(),
        log_every: arg(args, "--log-every", "25").parse().unwrap(),
    };
    let data = load_data(args);
    let mut rng = Rng::new(arg(args, "--seed", "0").parse().unwrap());
    let mut model = Model::new(cfg.clone(), &mut rng);
    println!(
        "ASCA v2: d_model {} n_heads {} n_layers {} n_experts {} d_ff {}",
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.n_experts, cfg.d_ff
    );
    train::train(
        &mut model,
        &data,
        &tcfg,
        arg(args, "--seed", "0").parse().unwrap(),
    );
    let vl = train::evaluate(&mut model, &data, &mut rng, &tcfg);
    println!("val_loss akhir: {:.4}  ppl {:.2}", vl, vl.exp());
    train::save(&out, &model).expect("gagal simpan checkpoint");
    println!("checkpoint disimpan: {out}");
}

fn cmd_eval(args: &[String]) {
    let ckpt = arg(args, "--ckpt", "");
    assert!(!arg(args, "--data", "").is_empty(), "butuh --data FILE");
    let (mut model, cfg) = train::load(&ckpt, 0).expect("gagal load checkpoint");
    let data = load_data(args);
    let tcfg = TrainCfg {
        steps: 0,
        seq_len: arg(args, "--seq-len", "128").parse().unwrap(),
        batch: 1,
        lr: 0.0,
        warmup: 0,
        eval_every: 0,
        eval_batches: arg(args, "--batches", "20").parse().unwrap(),
        log_every: 0,
    };
    let mut rng = Rng::new(1234);
    let vl = train::evaluate(&mut model, &data, &mut rng, &tcfg);
    println!(
        "ckpt d_model {} n_layers {} n_experts {}: val_loss {:.4}  ppl {:.2}",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_experts,
        vl,
        vl.exp()
    );
    let _ = Split::Train;
}

fn cmd_sample(args: &[String]) {
    use std::io::Write;
    let ckpt = arg(args, "--ckpt", "");
    let (model, cfg) = train::load(&ckpt, 0).expect("gagal load checkpoint");
    let tok = arg(args, "--tokenizer", "");
    let bpe = if tok.is_empty() {
        None
    } else {
        Some(Bpe::load(&tok).expect("gagal muat tokenizer"))
    };
    let prompt = arg(args, "--prompt", "Once upon a time");
    let n: usize = arg(args, "--tokens", "200").parse().unwrap();
    let temp: f32 = arg(args, "--temp", "0.8").parse().unwrap();
    let top_k: usize = arg(args, "--top-k", "0").parse().unwrap();
    let top_p: f32 = arg(args, "--top-p", "1.0").parse().unwrap();
    let mut rng = Rng::new(arg(args, "--seed", "0").parse().unwrap());

    // prompt -> id token (subword kalau ada tokenizer, byte kalau tidak)
    let mut ids: Vec<u32> = match &bpe {
        Some(b) => b.encode(prompt.as_bytes()),
        None => prompt.bytes().map(|c| c as u32).collect(),
    };
    if ids.is_empty() {
        ids.push(b'\n' as u32);
    }
    if ids.iter().any(|&i| i as usize >= cfg.vocab) {
        panic!("prompt berisi id di luar vocab {}", cfg.vocab);
    }

    // Decode inkremental: state dibawa antar token, jadi biaya per token konstan.
    // Versi lama menjalankan ulang seluruh prefix setiap token -> kuadratik.
    let mut st = model.new_decode_state();
    let mut logits = None;
    for &id in ids.iter() {
        logits = Some(model.step(id, &mut st));
    }
    let mut out_ids: Vec<u32> = ids.clone();
    for _ in 0..n {
        let lg = logits.as_ref().unwrap();
        let row = lg.row(0);
        let mut idx: Vec<usize> = (0..row.len()).collect();
        if top_k > 0 && top_k < row.len() {
            idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            idx.truncate(top_k);
        }
        let max = idx
            .iter()
            .map(|&i| row[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = idx
            .iter()
            .map(|&i| (i, ((row[i] - max) / temp).exp()))
            .collect();
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // top-p (nucleus)
        let total: f32 = probs.iter().map(|x| x.1).sum();
        let mut acc = 0.0;
        let mut cut = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            acc += p / total;
            if acc >= top_p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
        let sum: f32 = probs.iter().map(|x| x.1).sum();
        let mut r = rng.uniform() * sum;
        let mut pick = probs[0].0;
        for &(i, p) in probs.iter() {
            r -= p;
            if r <= 0.0 {
                pick = i;
                break;
            }
        }
        out_ids.push(pick as u32);
        logits = Some(model.step(pick as u32, &mut st));
    }

    // keluaran ditulis sebagai byte mentah supaya UTF-8 tidak rusak
    let bytes = match &bpe {
        Some(b) => b.decode(&out_ids),
        None => out_ids.iter().map(|&i| i as u8).collect(),
    };
    let mut so = std::io::stdout();
    let _ = so.write_all(&bytes);
    let _ = so.write_all(b"\n");
    let _ = so.flush();
}

fn cmd_gpu(args: &[String]) {
    use asca_rs::cuda;
    use asca_rs::tensor::{Mat, Rng};

    println!("{}", cuda::describe());
    if !cuda::available() {
        println!("tidak ada backend GPU; selesai.");
        return;
    }
    println!("kernel terdaftar: {:?}", cuda::KERNELS);

    // --- paritas terhadap CPU
    let mut rng = Rng::new(7);
    let mut worst: f32 = 0.0;
    for (m, k, n) in [(64, 64, 192), (128, 64, 512), (37, 53, 91), (256, 256, 256)] {
        let mut a = Mat::zeros(m, k);
        a.fill_rand(&mut rng, 0.1);
        let mut b = Mat::zeros(k, n);
        b.fill_rand(&mut rng, 0.1);
        cuda::set_enabled(false);
        let cpu = a.matmul(&b);
        cuda::set_enabled(true);
        let g = cuda::gpu().unwrap();
        let gpu_out = g.matmul(&a, &b).expect("matmul GPU gagal");
        let mut err = 0.0f32;
        let mut scale = 1e-9f32;
        for i in 0..cpu.d.len() {
            err = err.max((cpu.d[i] - gpu_out.d[i]).abs());
            scale = scale.max(cpu.d[i].abs());
        }
        let rel = err / scale;
        worst = worst.max(rel);
        println!("  matmul {m}x{k}@{k}x{n}: galat maks {err:.3e} (relatif {rel:.3e})");

        // transpose-A: C = A^T @ B dengan A disimpan (K,M) dan B (K,N) -> C (M,N)
        let mut at = Mat::zeros(k, m);
        at.fill_rand(&mut rng, 0.1);
        let mut b2 = Mat::zeros(k, n);
        b2.fill_rand(&mut rng, 0.1);
        cuda::set_enabled(false);
        let cpu_ta = at.transpose().matmul(&b2);
        cuda::set_enabled(true);
        let gpu_ta = g.matmul_ta(&at, &b2).expect("matmul_ta gagal");
        let mut e2 = 0.0f32;
        let mut s2 = 1e-9f32;
        for i in 0..cpu_ta.d.len() {
            e2 = e2.max((cpu_ta.d[i] - gpu_ta.d[i]).abs());
            s2 = s2.max(cpu_ta.d[i].abs());
        }
        println!(
            "  matmul_ta {k}x{m}^T@{k}x{n}: galat maks {e2:.3e} (relatif {:.3e})",
            e2 / s2
        );
        worst = worst.max(e2 / s2);
    }
    println!("galat relatif terburuk: {worst:.3e}");

    // --- paritas kernel delta rule terhadap referensi CPU
    {
        let (t_len, h, dh) = (24usize, 4usize, 16usize);
        let d = h * dh;
        let mut q = Mat::zeros(t_len, d);
        q.fill_rand(&mut rng, 1.0);
        let mut k = Mat::zeros(t_len, d);
        k.fill_rand(&mut rng, 1.0);
        let mut v = Mat::zeros(t_len, d);
        v.fill_rand(&mut rng, 0.5);
        // normalisasi L2 per head seperti di DeltaBlock
        for r in 0..t_len {
            for hh in 0..h {
                for mtx in [&mut q, &mut k] {
                    let row = mtx.row_mut(r);
                    let mut n2 = 0.0f32;
                    for c in 0..dh {
                        n2 += row[hh * dh + c] * row[hh * dh + c];
                    }
                    let inv = 1.0 / n2.sqrt().max(1e-12);
                    for c in 0..dh {
                        row[hh * dh + c] *= inv;
                    }
                }
            }
        }
        let alpha: Vec<f32> = (0..t_len * h).map(|_| 0.85 + 0.1 * rng.uniform()).collect();
        let beta: Vec<f32> = (0..t_len * h).map(|_| 0.7 + 0.2 * rng.uniform()).collect();
        // referensi CPU
        let mut cpu_out = Mat::zeros(t_len, d);
        let mut state = vec![0.0f32; h * dh * dh];
        for t in 0..t_len {
            for hh in 0..h {
                let off = hh * dh;
                let a = alpha[t * h + hh];
                let b = beta[t * h + hh];
                let mut sk = vec![0.0f32; dh];
                for i in 0..dh {
                    let mut acc = 0.0f32;
                    for a2 in 0..dh {
                        acc += state[hh * dh * dh + a2 * dh + i] * k.at(t, off + a2);
                    }
                    sk[i] = acc;
                }
                let mut u = vec![0.0f32; dh];
                for i in 0..dh {
                    u[i] = b * (v.at(t, off + i) - a * sk[i]);
                }
                for a2 in 0..dh {
                    for i in 0..dh {
                        let idx = hh * dh * dh + a2 * dh + i;
                        state[idx] = a * state[idx] + k.at(t, off + a2) * u[i];
                    }
                }
                for i in 0..dh {
                    let mut acc = 0.0f32;
                    for a2 in 0..dh {
                        acc += state[hh * dh * dh + a2 * dh + i] * q.at(t, off + a2);
                    }
                    cpu_out.set(t, off + i, acc);
                }
            }
        }
        let gpu_out = cuda::gpu()
            .unwrap()
            .delta_fwd(&q, &k, &v, &alpha, &beta, t_len, h, dh)
            .expect("delta_fwd gagal");
        let mut e = 0.0f32;
        let mut sc = 1e-9f32;
        for i in 0..cpu_out.d.len() {
            e = e.max((cpu_out.d[i] - gpu_out.d[i]).abs());
            sc = sc.max(cpu_out.d[i].abs());
        }
        println!(
            "  delta_fwd T={t_len} H={h} dh={dh}: galat maks {e:.3e} (relatif {:.3e})",
            e / sc
        );
    }

    // --- benchmark
    let sizes: Vec<(usize, usize, usize)> = arg(
        args,
        "--sizes",
        "128x64x192,512x64x1024,1024x1024x1024,4096x1024x1024",
    )
    .split(',')
    .filter_map(|s| {
        let p: Vec<usize> = s.split('x').filter_map(|x| x.trim().parse().ok()).collect();
        if p.len() == 3 {
            Some((p[0], p[1], p[2]))
        } else {
            None
        }
    })
    .collect();
    println!("\n| M | K | N | CPU ms | GPU ms | CPU GFLOPS | GPU GFLOPS | speedup |");
    println!("|---|---|---|---|---|---|---|---|");
    for (m, k, n) in sizes {
        let mut a = Mat::zeros(m, k);
        a.fill_rand(&mut rng, 0.1);
        let mut b = Mat::zeros(k, n);
        b.fill_rand(&mut rng, 0.1);
        let reps = if m * k * n > 100_000_000 { 3 } else { 20 };

        cuda::set_enabled(false);
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(a.matmul(&b));
        }
        let cpu_s = t0.elapsed().as_secs_f32() / reps as f32;

        let g = cuda::gpu().unwrap();
        let _ = g.matmul(&a, &b);
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(g.matmul(&a, &b).unwrap());
        }
        let gpu_s = t1.elapsed().as_secs_f32() / reps as f32;
        cuda::set_enabled(true);

        let flops = 2.0 * (m * k * n) as f64;
        println!(
            "| {m} | {k} | {n} | {:.3} | {:.3} | {:.1} | {:.1} | {:.2}x |",
            cpu_s * 1e3,
            gpu_s * 1e3,
            flops / cpu_s as f64 / 1e9,
            flops / gpu_s as f64 / 1e9,
            cpu_s / gpu_s
        );
    }
}

fn cmd_bench(args: &[String]) {
    use asca_rs::bench::{self, BenchCfg, Trainable};
    use asca_rs::dense::dense_matched_to;

    assert!(!arg(args, "--data", "").is_empty(), "butuh --data FILE");
    let cfg = build_cfg(args);
    let bcfg = BenchCfg {
        steps: arg(args, "--steps", "400").parse().unwrap(),
        seq_len: arg(args, "--seq-len", "128").parse().unwrap(),
        batch: arg(args, "--batch", "4").parse().unwrap(),
        lr: arg(args, "--lr", "3e-3").parse().unwrap(),
        warmup: arg(args, "--warmup", "40").parse().unwrap(),
        eval_batches: arg(args, "--eval-batches", "8").parse().unwrap(),
    };
    let seeds: Vec<u64> = arg(args, "--seeds", "0,1,2")
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let eval_every: usize = arg(args, "--eval-curve", "0").parse().unwrap();
    // Aturan proyek: LR harus disetel per arsitektur. Tanpa override ini, satu LR dipakai
    // untuk keduanya, dan baseline dense bisa kalah hanya karena LR-nya tidak cocok.
    let mut bcfg_dense = bcfg.clone();
    bcfg_dense.lr = arg(args, "--lr-dense", &bcfg.lr.to_string())
        .parse()
        .unwrap();
    if bcfg_dense.lr != bcfg.lr {
        println!("lr: asca {} | dense {}", bcfg.lr, bcfg_dense.lr);
    }
    let data = load_data(args);
    println!(
        "bench: steps {} seq {} batch {} lr {} seeds {:?}",
        bcfg.steps, bcfg.seq_len, bcfg.batch, bcfg.lr, seeds
    );
    println!(
        "data: {} byte -> {} token ({} train, {:.2} byte/token)",
        data.raw_bytes,
        data.tokens.len(),
        data.train_end,
        data.raw_bytes as f64 / data.tokens.len() as f64
    );

    // acuan: ASCA seed pertama, untuk menentukan target pencocokan parameter
    let mut rng0 = Rng::new(seeds[0]);
    let asca_ref = Model::new(cfg.clone(), &mut rng0);
    let asca_total = asca_ref.num_params();
    let asca_active = asca_ref.active_params();
    println!(
        "ASCA: total {} params, aktif/token {}",
        asca_total, asca_active
    );

    // Dua perbandingan yang sah: total params sama, dan active params sama.
    let want = arg(args, "--match", "both");
    let modes: Vec<(&str, usize)> = [("total", asca_total), ("active", asca_active)]
        .into_iter()
        .filter(|(m, _)| want == "both" || want == *m)
        .collect();
    // Sweep per arsitektur: menjalankan baseline dense saat hanya LR ASCA yang disapu
    // membuang ~90% waktu (dense total-matched ~75 s per 400 langkah).
    let only = arg(args, "--only", "both");
    for (mode, target) in modes {
        println!("\n=== pencocokan {mode} params (target {target}) ===");
        let mut a_losses = Vec::new();
        let mut d_losses = Vec::new();
        let mut a_tps = Vec::new();
        let mut d_tps = Vec::new();
        let (mut dp, mut da, mut ds) = (0usize, 0usize, 0usize);
        let mut d_ff_used = 0usize;
        for &seed in seeds.iter() {
            let mut rng = Rng::new(seed);
            let mut asca = Model::new(cfg.clone(), &mut rng);
            let (a_loss, a_tps_v, curve_a) = if only == "dense" {
                (0.0f32, 0.0f32, Vec::new())
            } else {
                let (r, ca) = bench::run_curve("asca", &mut asca, &data, &bcfg, seed, eval_every);
                (r.val_loss, r.tok_per_sec, ca)
            };
            if only == "asca" {
                println!("  seed {seed}: asca {a_loss:.4} ({a_tps_v:.0} tok/s)");
                a_losses.push(a_loss);
                a_tps.push(a_tps_v);
                continue;
            }
            let mut dense = dense_matched_to(
                target,
                cfg.vocab,
                cfg.d_model,
                cfg.n_heads,
                cfg.n_layers,
                seed ^ 0x5eed,
            );
            let (rd, curve_d) =
                bench::run_curve("dense", &mut dense, &data, &bcfg_dense, seed, eval_every);
            if eval_every > 0 {
                println!("    kurva (token, asca, dense):");
                for ((tok, la), (_, ld)) in curve_a.iter().zip(curve_d.iter()) {
                    println!(
                        "      {:>8}  {:.4}  {:.4}  delta {:+.4}",
                        tok,
                        la,
                        ld,
                        la - ld
                    );
                }
            }
            println!(
                "  seed {seed}: asca {:.4} ({:.0} tok/s) | dense d_ff={} {:.4} ({:.0} tok/s)",
                a_loss, a_tps_v, dense.cfg.d_ff, rd.val_loss, rd.tok_per_sec
            );
            a_losses.push(a_loss);
            d_losses.push(rd.val_loss);
            a_tps.push(a_tps_v);
            d_tps.push(rd.tok_per_sec);
            dp = rd.params;
            da = rd.active_params;
            ds = rd.decode_floats;
            d_ff_used = dense.cfg.d_ff;
        }
        let (am, asd) = bench::mean_std(&a_losses);
        let (dm, dsd) = bench::mean_std(&d_losses);
        let (at, _) = bench::mean_std(&a_tps);
        let (dt, _) = bench::mean_std(&d_tps);
        println!();
        println!("| model | params | aktif/token | val_loss | tok/s | state decode |");
        println!("|---|---|---|---|---|---|");
        println!(
            "| ASCA v2 | {} | {} | {:.4} ± {:.4} | {:.0} | {} float |",
            asca_total,
            asca_active,
            am,
            asd,
            at,
            asca_ref.decode_state_floats(bcfg.seq_len)
        );
        println!(
            "| dense (d_ff={}) | {} | {} | {:.4} ± {:.4} | {:.0} | {} float |",
            d_ff_used, dp, da, dm, dsd, dt, ds
        );
        println!(
            "delta val_loss (asca - dense) = {:+.4}   throughput asca/dense = {:.2}x",
            am - dm,
            at / dt
        );
    }
}

fn cmd_inspect(args: &[String]) {
    let ckpt = arg(args, "--ckpt", "");
    let (model, cfg) = train::load(&ckpt, 0).expect("gagal load checkpoint");
    println!("checkpoint: {ckpt}");
    println!(
        "config: vocab {} d_model {} n_heads {} n_layers {} n_experts {} d_ff {}",
        cfg.vocab, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.n_experts, cfg.d_ff
    );
    println!("params: {}", model.num_params());
    for p in model.params() {
        println!(
            "  {:<16} {:>8} x {:<8} = {}",
            p.name,
            p.value.rows,
            p.value.cols,
            p.numel()
        );
    }
}
