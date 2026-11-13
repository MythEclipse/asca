//! CLI ASCA v2: train / eval / sample / inspect.

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
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).cloned().unwrap_or_else(|| "help".into());
    match cmd.as_str() {
        "train" => cmd_train(&args),
        "eval" => cmd_eval(&args),
        "sample" => cmd_sample(&args),
        "inspect" => cmd_inspect(&args),
        "bench" => cmd_bench(&args),
        "gpu" => cmd_gpu(&args),
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
}

fn build_cfg(args: &[String]) -> Config {
    Config {
        vocab: 256,
        d_model: arg(args, "--d-model", "64").parse().unwrap(),
        n_heads: arg(args, "--n-heads", "4").parse().unwrap(),
        n_layers: arg(args, "--n-layers", "2").parse().unwrap(),
        n_experts: arg(args, "--n-experts", "8").parse().unwrap(),
        d_ff: arg(args, "--d-ff", "128").parse().unwrap(),
    }
}

fn cmd_train(args: &[String]) {
    let data_path = arg(args, "--data", "");
    assert!(!data_path.is_empty(), "butuh --data FILE");
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
    let data = Data::load(&data_path, 0.05).expect("gagal baca data");
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
    let data_path = arg(args, "--data", "");
    let (mut model, cfg) = train::load(&ckpt, 0).expect("gagal load checkpoint");
    let data = Data::load(&data_path, 0.05).expect("gagal baca data");
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
    let ckpt = arg(args, "--ckpt", "");
    let (model, _cfg) = train::load(&ckpt, 0).expect("gagal load checkpoint");
    let prompt = arg(args, "--prompt", "Once upon a time");
    let n: usize = arg(args, "--tokens", "200").parse().unwrap();
    let mut ctx: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
    if ctx.is_empty() {
        ctx.push(b'\n' as u32);
    }
    let mut rng = Rng::new(arg(args, "--seed", "0").parse().unwrap());
    print!("{prompt}");
    for _ in 0..n {
        let (logits, _) = model.forward(&ctx);
        let last = logits.rows - 1;
        let row = logits.row(last);
        // sampling suhu 0.8
        let temp = 0.8f32;
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| ((v - max) / temp).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut r = rng.uniform() * sum;
        let mut pick = 0usize;
        for (i, &e) in exps.iter().enumerate() {
            r -= e;
            if r <= 0.0 {
                pick = i;
                break;
            }
        }
        ctx.push(pick as u32);
        print!("{}", pick as u8 as char);
        if ctx.len() > 512 {
            ctx.drain(0..ctx.len() - 512);
        }
    }
    println!();
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

    let data_path = arg(args, "--data", "");
    assert!(!data_path.is_empty(), "butuh --data FILE");
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
    let data = Data::load(&data_path, 0.05).expect("gagal baca data");
    println!(
        "bench: steps {} seq {} batch {} lr {} seeds {:?}",
        bcfg.steps, bcfg.seq_len, bcfg.batch, bcfg.lr, seeds
    );
    println!("data: {} byte ({} train)", data.bytes.len(), data.train_end);

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
            let (r, curve_a) = bench::run_curve("asca", &mut asca, &data, &bcfg, seed, eval_every);
            let mut dense = dense_matched_to(
                target,
                cfg.vocab,
                cfg.d_model,
                cfg.n_heads,
                cfg.n_layers,
                seed ^ 0x5eed,
            );
            let (rd, curve_d) =
                bench::run_curve("dense", &mut dense, &data, &bcfg, seed, eval_every);
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
                r.val_loss, r.tok_per_sec, dense.cfg.d_ff, rd.val_loss, rd.tok_per_sec
            );
            a_losses.push(r.val_loss);
            d_losses.push(rd.val_loss);
            a_tps.push(r.tok_per_sec);
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
