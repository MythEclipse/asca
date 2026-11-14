//! CLI ASCA v2: train / eval / sample / inspect.

use asca_rs::bpe::Bpe;
use asca_rs::dense::{DenseCfg, DenseModel};
use asca_rs::model::{Config, Model};
use asca_rs::tensor::{AdamW, Rng};
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
        "tokenize" => cmd_tokenize(&args),
        "chat" => cmd_chat(&args),
        "genstat" => cmd_genstat(&args),
        "ckptinfo" => cmd_ckptinfo(&args),
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
    // File token ter-encode menang: tidak perlu meng-encode ulang korpus.
    let tokfile = arg(args, "--tokenized", "");
    if !tokfile.is_empty() {
        return Data::load_tokens(&tokfile, 0.05).expect("gagal muat file token");
    }
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
    let tokfile = arg(args, "--tokenized", "");
    if !tokfile.is_empty() {
        return Data::vocab_of_tokens(&tokfile).expect("gagal baca header file token");
    }
    let tok = arg(args, "--tokenizer", "");
    if tok.is_empty() {
        arg(args, "--vocab", "256").parse().unwrap()
    } else {
        Bpe::load(&tok).expect("gagal muat tokenizer").vocab_size()
    }
}

/// Encode korpus sekali, simpan id token. Dipakai oleh --tokenized.
fn cmd_tokenize(args: &[String]) {
    let path = arg(args, "--data", "");
    let tok = arg(args, "--tokenizer", "");
    let out = arg(args, "--out", "corpus.tok");
    let bpe = Bpe::load(&tok).expect("gagal muat tokenizer");
    let t0 = std::time::Instant::now();
    let data = Data::load_with_bpe(&path, 0.05, &bpe).expect("gagal baca data");
    data.save_tokens(&out, bpe.vocab_size())
        .expect("gagal simpan token");
    println!(
        "tokenize: {} byte -> {} token ({:.2} byte/token) dalam {:.1} s -> {out}",
        data.raw_bytes,
        data.tokens.len(),
        data.raw_bytes as f64 / data.tokens.len() as f64,
        t0.elapsed().as_secs_f64()
    );
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
    assert!(
        !arg(args, "--data", "").is_empty() || !arg(args, "--tokenized", "").is_empty(),
        "butuh --data FILE atau --tokenized FILE"
    );
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
    // --resume melanjutkan bobot + momen AdamW + nomor langkah, jadi run panjang bisa
    // dihentikan kapan saja tanpa kehilangan progres.
    let resume = arg(args, "--resume", "");
    let (mut model, mut opt, start_step) = if resume.is_empty() {
        let m = Model::new(cfg.clone(), &mut rng);
        (m, AdamW::new(tcfg.lr), 0usize)
    } else {
        let (m, o, s) = train::load_resume(&resume).expect("gagal load resume");
        println!("resume: {resume} dari langkah {s}");
        (m, o, s)
    };
    println!(
        "ASCA v2: d_model {} n_heads {} n_layers {} n_experts {} d_ff {}",
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.n_experts, cfg.d_ff
    );
    opt.lr = tcfg.lr;
    let out2 = out.clone();
    train::train_resume(
        &mut model,
        &data,
        &tcfg,
        arg(args, "--seed", "0").parse().unwrap(),
        opt,
        start_step,
        move |m, _o, s| train::save_resume(&out2, m, s),
    )
    .expect("gagal training");
    let vl = train::evaluate(&mut model, &data, &mut rng, &tcfg);
    println!(
        "val_loss akhir: {:.4}  ppl {:.2}  bpb {:.4}",
        vl,
        vl.exp(),
        data.bits_per_byte(vl)
    );
    train::save_resume(&out, &model, tcfg.steps).expect("gagal simpan checkpoint");
    println!("checkpoint disimpan: {out}");
}

fn cmd_eval(args: &[String]) {
    let ckpt = arg(args, "--ckpt", "");
    assert!(
        !arg(args, "--data", "").is_empty() || !arg(args, "--tokenized", "").is_empty(),
        "butuh --data FILE atau --tokenized FILE"
    );
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
        "ckpt d_model {} n_layers {} n_experts {}: val_loss {:.4}  ppl {:.2}  bpb {:.4}",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_experts,
        vl,
        vl.exp(),
        data.bits_per_byte(vl)
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

/// REPL untuk benar-benar memakai model: satu baris masukan, jawaban mengalir token demi
/// token. Memakai jalur decode inkremental yang sama dengan `sample`, jadi biaya per token
/// konstan dan konteks percakapan bisa terus tumbuh tanpa melambat.
/// Panjang awalan `b` yang valid sebagai UTF-8. Dipakai untuk menahan byte terakhir yang
/// mungkin masih separuh karakter, sekaligus menahan calon penanda berhenti.
fn utf8_prefix_len(b: &[u8]) -> usize {
    for cut in (0..=b.len()).rev() {
        if std::str::from_utf8(&b[..cut]).is_ok() {
            return cut;
        }
    }
    0
}

/// Penanda batas cerita di korpus. Model mempelajarinya dan akan menghasilkannya sendiri;
/// tanpa ini, penanda itu ikut tercetak sebagai "<|endoftext|>" di tengah jawaban.
const STOP_MARK: &[u8] = b"<|endoftext|>";

/// Cetak isi header checkpoint tanpa memuat bobotnya: nomor langkah dan konfigurasi model.
/// Dipakai `scripts/continue_training.sh` untuk menghitung learning rate sambungan.
fn cmd_ckptinfo(args: &[String]) {
    let path = arg(args, "--ckpt", "");
    let (model, step) = match train::peek_resume(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("gagal membaca header checkpoint: {e}");
            std::process::exit(1);
        }
    };
    println!("step {step}");
    println!("vocab {}", model.vocab);
    println!("d_model {}", model.d_model);
    println!("n_heads {}", model.n_heads);
    println!("n_layers {}", model.n_layers);
    println!("n_experts {}", model.n_experts);
    println!("d_ff {}", model.d_ff);
}

fn cmd_chat(args: &[String]) {
    use std::io::{BufRead, Write};
    let ckpt = arg(args, "--ckpt", "");
    let (model, cfg) = train::load(&ckpt, 0).expect("gagal load checkpoint");
    let tok = arg(args, "--tokenizer", "");
    let bpe = if tok.is_empty() {
        None
    } else {
        Some(Bpe::load(&tok).expect("gagal muat tokenizer"))
    };
    let mut temp: f32 = arg(args, "--temp", "0.8").parse().unwrap();
    let top_k: usize = arg(args, "--top-k", "40").parse().unwrap();
    let top_p: f32 = arg(args, "--top-p", "0.95").parse().unwrap();
    let max_new: usize = arg(args, "--tokens", "200").parse().unwrap();
    let mut rng = Rng::new(arg(args, "--seed", "0").parse().unwrap());
    let mut st = model.new_decode_state();
    let mut ctx_ids: Vec<u32> = Vec::new();
    println!(
        "ASCA chat — /reset, /temp N, /quit. Model {} param.",
        model.num_params()
    );
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }
        if line == "/reset" {
            ctx_ids.clear();
            st = model.new_decode_state();
            println!("(konteks direset)");
            continue;
        }
        if let Some(rest) = line.strip_prefix("/temp ") {
            if let Ok(v) = rest.trim().parse() {
                temp = v;
                println!("(temp = {temp})");
            }
            continue;
        }
        let mut ids: Vec<u32> = match &bpe {
            Some(b) => b.encode(line.as_bytes()),
            None => line.bytes().map(|c| c as u32).collect(),
        };
        ids.retain(|&i| (i as usize) < cfg.vocab);
        if ids.is_empty() {
            continue;
        }
        // prompt masuk ke state, lalu jawaban mengalir keluar
        let mut logits = None;
        // Pemisah dokumen antar giliran. Model dilatih dengan `<|endoftext|>` sebagai batas
        // cerita; tanpa penanda ini, prompt baru dianggap lanjutan cerita sebelumnya dan
        // model menjawab dengan mengulang tokoh lama (terlihat sebagai " Tom. He was so
        // excited!" padahal promptnya tentang kotak di taman). Menyisipkan penanda yang
        // sama seperti saat training membuat tiap giliran menjadi dokumen baru.
        if !ctx_ids.is_empty() {
            let sep = match &bpe {
                Some(b) => b.encode(STOP_MARK),
                None => STOP_MARK.iter().map(|&c| c as u32).collect(),
            };
            for &id in sep.iter() {
                if (id as usize) < cfg.vocab {
                    logits = Some(model.step(id, &mut st));
                    ctx_ids.push(id);
                }
            }
        }
        for &id in ctx_ids.iter().chain(ids.iter()) {
            logits = Some(model.step(id, &mut st));
        }
        ctx_ids.extend_from_slice(&ids);
        let mut out_bytes: Vec<u8> = Vec::new();
        let mut produced: Vec<u32> = Vec::new();
        for _ in 0..max_new {
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
            produced.push(pick as u32);
            ctx_ids.push(pick as u32);
            // cetak hanya byte yang sudah lengkap (subword bisa memotong UTF-8)
            let chunk = match &bpe {
                Some(b) => b.decode(&[pick as u32]),
                None => vec![pick as u8],
            };
            out_bytes.extend_from_slice(&chunk);
            let mut stop = false;
            if let Some(p) = out_bytes
                .windows(STOP_MARK.len())
                .position(|w| w == STOP_MARK)
            {
                out_bytes.truncate(p);
                stop = true;
            }
            // Tahan byte terakhir yang mungkin masih separuh karakter atau calon awal
            // penanda berhenti, supaya tidak ada karakter rusak yang tercetak.
            let hold = if stop { 0 } else { STOP_MARK.len() - 1 };
            let upto = out_bytes.len().saturating_sub(hold);
            let n = utf8_prefix_len(&out_bytes[..upto]);
            if n > 0 {
                print!("{}", String::from_utf8_lossy(&out_bytes[..n]));
                let _ = std::io::stdout().flush();
                out_bytes.drain(..n);
            }
            if stop {
                break;
            }
            logits = Some(model.step(pick as u32, &mut st));
        }
        println!();
    }
}

/// Ukur kualitas generasi tanpa perlu penilaian manusia.
///
/// Val loss hanya mengukur seberapa baik model menebak token berikutnya pada teks yang
/// diberikan; ia tidak menangkap mode collapse (model yang mengulang kalimat yang sama
/// tetap punya loss rendah). Karena itu di sini diukur dua hal yang saling melengkapi:
///
///   distinct-n : rasio n-gram unik terhadap total n-gram. Turun = model mulai mengulang.
///   rep-4      : 1 - distinct-4, yaitu laju pengulangan 4-gram.
///   kata/kalim : panjang kalimat rata-rata, penanda apakah model berhenti menulis
///                potongan pendek.
///
/// Semua diukur pada keluaran model sendiri dengan prompt tetap, sehingga bisa
/// dibandingkan antar checkpoint.
fn cmd_genstat(args: &[String]) {
    let ckpt = arg(args, "--ckpt", "");
    let (model, cfg) = train::load(&ckpt, 0).expect("gagal load checkpoint");
    let tokpath = arg(args, "--tokenizer", "");
    let bpe = Bpe::load(&tokpath).expect("gagal muat tokenizer");
    let n_samples: usize = arg(args, "--samples", "8").parse().unwrap();
    let max_new: usize = arg(args, "--tokens", "120").parse().unwrap();
    let temp: f32 = arg(args, "--temp", "0.8").parse().unwrap();
    let top_p: f32 = arg(args, "--top-p", "0.95").parse().unwrap();
    let prompts = [
        "Once upon a time, there was a little girl named Lily",
        "One day, a small dog found a big red ball",
        "The sun was shining and the children went outside",
        "Tom was sad because he lost his toy",
    ];
    let mut rng = Rng::new(7);
    let mut per_sample: Vec<Vec<String>> = Vec::new();
    let mut gen_ids: Vec<Vec<u32>> = Vec::new();
    for si in 0..n_samples {
        let mut produced: Vec<u32> = Vec::new();
        let prompt = prompts[si % prompts.len()];
        let mut ids = bpe.encode(prompt.as_bytes());
        ids.retain(|&i| (i as usize) < cfg.vocab);
        let mut st = model.new_decode_state();
        let mut logits = None;
        for &id in ids.iter() {
            logits = Some(model.step(id, &mut st));
        }
        let mut text = String::new();
        for _ in 0..max_new {
            let lg = logits.as_ref().unwrap();
            let row = lg.row(0);
            let v = row.len();
            let mut idx: Vec<usize> = (0..v).collect();
            let k = 40.min(v);
            idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            idx.truncate(k);
            let max = idx
                .iter()
                .map(|&i| row[i])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<(usize, f32)> = idx
                .iter()
                .map(|&i| (i, ((row[i] - max) / temp).exp()))
                .collect();
            probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
            produced.push(pick as u32);
            let chunk = bpe.decode(&[pick as u32]);
            if let Ok(s) = std::str::from_utf8(&chunk) {
                text.push_str(s);
            }
            logits = Some(model.step(pick as u32, &mut st));
        }
        // Buang penanda batas cerita supaya tidak terhitung sebagai kata.
        let text = match text.find("<|endoftext|>") {
            Some(p) => text[..p].to_string(),
            None => text,
        };
        gen_ids.push(produced);
        per_sample.push(
            text.split_whitespace()
                .map(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                })
                .filter(|w| !w.is_empty())
                .collect(),
        );
    }
    // Dihitung PER SAMPEL lalu dirata-rata. Versi pertama menggabungkan semua sampel jadi
    // satu daftar kata, sehingga akhir cerita yang formulaik ("they had a great time")
    // yang muncul di banyak sampel terhitung sebagai pengulangan - padahal itu bukan
    // model mengulang dalam satu cerita, melainkan korpusnya memang bertemplat. Metrik
    // yang mencampur keduanya tidak bisa membedakan mode collapse dari gaya korpus.
    let distinct_in = |words: &[String], n: usize| -> f32 {
        if words.len() < n {
            return 0.0;
        }
        let mut set = std::collections::HashSet::new();
        let mut count = 0usize;
        for w in words.windows(n) {
            set.insert(w.join(" "));
            count += 1;
        }
        if count == 0 {
            return 0.0;
        }
        set.len() as f32 / count as f32
    };
    let avg = |n: usize| -> f32 {
        if per_sample.is_empty() {
            return 0.0;
        }
        per_sample.iter().map(|w| distinct_in(w, n)).sum::<f32>() / per_sample.len() as f32
    };
    let total: usize = per_sample.iter().map(|w| w.len()).sum();
    let mut vocab = std::collections::HashSet::new();
    for w in per_sample.iter().flatten() {
        vocab.insert(w.clone());
    }
    println!("genstat: {n_samples} sampel x {max_new} token (temp {temp}, top-p {top_p})");
    println!(
        "  kata total   : {total} (rata-rata {:.0}/sampel)",
        total as f32 / n_samples as f32
    );
    println!("  distinct-1   : {:.4}", avg(1));
    println!("  distinct-2   : {:.4}", avg(2));
    println!("  distinct-3   : {:.4}", avg(3));
    println!("  rep-4        : {:.4}", 1.0 - avg(4));
    println!("  vocab unik   : {} (gabungan semua sampel)", vocab.len());

    // Self-bpb: seberapa mudah model memprediksi keluarannya SENDIRI.
    //
    // distinct-n dan rep-4 hanya mengukur keragaman, dan keragaman selalu naik kalau suhu
    // dinaikkan - teks acak punya keragaman maksimal. Jadi angka itu tidak bisa dipakai
    // memilih suhu. Self-bpb adalah pasangannya: teks yang runtut mudah ditebak model,
    // teks ngawur tidak. Kualitas generasi adalah keduanya sekaligus, bukan salah satu.
    let mut loss_sum = 0.0f64;
    let mut tok_sum = 0usize;
    for ids in gen_ids.iter() {
        if ids.len() < 2 {
            continue;
        }
        loss_sum += model.loss(&ids[..ids.len() - 1], &ids[1..]) as f64 * (ids.len() - 1) as f64;
        tok_sum += ids.len() - 1;
    }
    if tok_sum > 0 {
        let nats = loss_sum / tok_sum as f64;
        println!(
            "  self-bpb     : {:.4} (nats/token {:.4})",
            nats / std::f64::consts::LN_2,
            nats
        );
    }
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

    // --- paritas kernel delta rule terhadap referensi CPU (batched)
    //
    // Diuji dengan batch > 1 karena di situlah bug paling mudah terjadi: kalau satu blok
    // menelusuri seluruh `B*seq` baris sebagai satu sekuens, hasilnya tetap "masuk akal"
    // tapi state bocor antar sekuens. Referensi CPU di bawah me-reset state di awal tiap
    // sekuens, jadi kebocoran itu akan terlihat sebagai galat besar.
    {
        let (batch, seq, h, dh) = (3usize, 24usize, 4usize, 16usize);
        let t_len = batch * seq;
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
        let mut alpha = Mat::zeros(t_len, h);
        let mut beta = Mat::zeros(t_len, h);
        for r in 0..t_len {
            for hh in 0..h {
                alpha.set(r, hh, 0.85 + 0.1 * rng.uniform());
                beta.set(r, hh, 0.7 + 0.2 * rng.uniform());
            }
        }
        // referensi CPU: state di-reset per sekuens
        let mut cpu_out = Mat::zeros(t_len, d);
        let mut cpu_states: Vec<Vec<f32>> = Vec::with_capacity(h * batch);
        for hh in 0..h {
            for bi in 0..batch {
                let off = hh * dh;
                let mut state = vec![0.0f32; dh * dh];
                let mut hist: Vec<f32> = Vec::with_capacity((seq + 1) * dh * dh);
                for t in 0..seq {
                    let r = bi * seq + t;
                    hist.extend_from_slice(&state);
                    let a = alpha.at(r, hh);
                    let b = beta.at(r, hh);
                    let mut sk = vec![0.0f32; dh];
                    for i in 0..dh {
                        let mut acc = 0.0f32;
                        for a2 in 0..dh {
                            acc += state[a2 * dh + i] * k.at(r, off + a2);
                        }
                        sk[i] = acc;
                    }
                    let mut u = vec![0.0f32; dh];
                    for i in 0..dh {
                        u[i] = b * (v.at(r, off + i) - a * sk[i]);
                    }
                    for a2 in 0..dh {
                        for i in 0..dh {
                            state[a2 * dh + i] = a * state[a2 * dh + i] + k.at(r, off + a2) * u[i];
                        }
                    }
                    for i in 0..dh {
                        let mut acc = 0.0f32;
                        for a2 in 0..dh {
                            acc += state[a2 * dh + i] * q.at(r, off + a2);
                        }
                        cpu_out.set(r, off + i, acc);
                    }
                }
                hist.extend_from_slice(&state);
                cpu_states.push(hist);
            }
        }
        let (gpu_out, dev) = cuda::gpu()
            .unwrap()
            .delta_scan(&q, &k, &v, &alpha, &beta, batch, seq, h, dh)
            .expect("delta_scan gagal");
        let gpu_states = dev.download_states().expect("unduh state gagal");
        let mut e = 0.0f32;
        let mut sc = 1e-9f32;
        for i in 0..cpu_out.d.len() {
            e = e.max((cpu_out.d[i] - gpu_out.d[i]).abs());
            sc = sc.max(cpu_out.d[i].abs());
        }
        let mut se = 0.0f32;
        let mut ssc = 1e-9f32;
        for (a, b) in cpu_states.iter().zip(gpu_states.iter()) {
            for i in 0..a.len() {
                se = se.max((a[i] - b[i]).abs());
                ssc = ssc.max(a[i].abs());
            }
        }
        println!(
            "  delta_scan B={batch} seq={seq} H={h} dh={dh}: o galat {e:.3e} (relatif {:.3e}), states galat {se:.3e} (relatif {:.3e})",
            e / sc,
            se / ssc
        );

        // Isolasi antar sekuens: hasil batch harus sama dengan memanggil tiap sekuens
        // sendiri-sendiri. Ini yang membuktikan state tidak bocor.
        let mut leak = 0.0f32;
        for bi in 0..batch {
            let lo = bi * seq;
            let hi = lo + seq;
            let sub = |m: &Mat| Mat::from_vec(seq, d, m.d[lo * d..hi * d].to_vec());
            let (single, _) = cuda::gpu()
                .unwrap()
                .delta_scan(
                    &sub(&q),
                    &sub(&k),
                    &sub(&v),
                    &Mat::from_vec(seq, h, alpha.d[lo * h..hi * h].to_vec()),
                    &Mat::from_vec(seq, h, beta.d[lo * h..hi * h].to_vec()),
                    1,
                    seq,
                    h,
                    dh,
                )
                .expect("delta_scan (batch=1) gagal");
            for t in 0..seq {
                for c in 0..d {
                    leak = leak.max((single.at(t, c) - gpu_out.at(lo + t, c)).abs());
                }
            }
        }
        println!("  delta_scan isolasi sekuens: galat batch={batch} vs {batch}x batch=1 {leak:.3e}");
    }

    // --- paritas MoE device-resident vs jalur CPU
    //
    // Jalur MoE memakai gather -> GEMM (offset kolom) -> bias -> SiLU -> GEMM -> scatter
    // di device. Yang mudah salah di situ: offset kolom `w1`, offset baris `w2`, dan
    // bias per expert. Gate ini membandingkan keluaran blok plus cache-nya (`pre`,
    // `hidden`) dengan jalur CPU yang sudah diverifikasi gradcheck.
    {
        use asca_rs::model::HashMoe;
        let (t_len, d, d_ff, n_experts) = (1024usize, 192usize, 384usize, 8usize);
        let mut rng_a = Rng::new(77);
        let mut moe_cpu = HashMoe::new(d, d_ff, n_experts, &mut rng_a);
        let mut rng_b = Rng::new(77);
        let mut moe_gpu = HashMoe::new(d, d_ff, n_experts, &mut rng_b);
        let mut x = Mat::zeros(t_len, d);
        x.fill_rand(&mut rng, 0.5);
        let ids: Vec<u32> = (0..t_len).map(|_| rng.below(4096) as u32).collect();
        cuda::set_enabled(false);
        let (out_cpu, c_cpu) = moe_cpu.forward(&x, &ids);
        cuda::set_enabled(true);
        let (out_gpu, c_gpu) = moe_gpu.forward(&x, &ids);
        let rel = |a: &[f32], b: &[f32]| {
            let mut e = 0.0f32;
            let mut sc = 1e-9f32;
            for i in 0..a.len() {
                e = e.max((a[i] - b[i]).abs());
                sc = sc.max(a[i].abs());
            }
            e / sc
        };
        println!(
            "  moe device: out rel {:.3e}, pre rel {:.3e}, hidden rel {:.3e}",
            rel(&out_cpu.d, &out_gpu.d),
            rel(&c_cpu.pre, &c_gpu.pre),
            rel(&c_cpu.hidden, &c_gpu.hidden)
        );
    }

    // --- paritas kepala output device-resident vs jalur host
    //
    // Yang diuji bukan cuma angkanya, tapi bahwa memindahkan logits ke VRAM tidak
    // mengubah loss maupun gradien: `dlogits` yang dulu naik-turun 4x harus menghasilkan
    // d_head dan dh yang sama persis dengan jalur yang mengunduhnya.
    {
        let (t_len, d, v) = (512usize, 192usize, 12288usize);
        let mut h_final = Mat::zeros(t_len, d);
        h_final.fill_rand(&mut rng, 0.5);
        let mut head = Mat::zeros(d, v);
        head.fill_rand(&mut rng, 0.05);
        let targets: Vec<u32> = (0..t_len).map(|_| (rng.uniform() * v as f32) as u32).collect();
        let g = cuda::gpu().unwrap();
        // jalur device
        let dev = g.head_forward_dev(&h_final, &head).expect("head fwd dev");
        let head_t = head.transpose();
        let (loss_dev, d_head_dev, dh_dev) = g
            .head_bwd_dev(&dev, &targets, &head_t)
            .expect("head bwd dev");
        // jalur host (logits turun, CE, dua GEMM)
        let logits = h_final.matmul(&head);
        let (loss_host, dlogits) = g.cross_entropy_bwd(&logits, &targets).expect("CE");
        let d_head_host = h_final.matmul_ta(&dlogits);
        let dh_host = dlogits.matmul_t(&head);
        let rel = |a: &Mat, b: &Mat| {
            let mut e = 0.0f32;
            let mut sc = 1e-9f32;
            for i in 0..a.d.len() {
                e = e.max((a.d[i] - b.d[i]).abs());
                sc = sc.max(a.d[i].abs());
            }
            e / sc
        };
        println!(
            "  head device: loss {loss_dev:.6} vs {loss_host:.6} (selisih {:.2e}), d_head rel {:.3e}, dh rel {:.3e}",
            (loss_dev - loss_host).abs(),
            rel(&d_head_dev, &d_head_host),
            rel(&dh_dev, &dh_host)
        );
        println!(
            "  head device: H2D+D2H {:.1} MB vs jalur host {:.1} MB per langkah",
            (t_len * d * 4 + d * v * 4 + d * v * 4 + t_len * d * 4) as f64 / 1e6,
            (t_len * v * 4 * 2 + t_len * v * 4 * 2 + d * v * 4 + t_len * d * 4) as f64 / 1e6
        );
    }

    // --- paritas gradien blok delta: BPTT di GPU vs BPTT di CPU
    //
    // Ini gate yang sebenarnya untuk `delta_bwd_f32`: kernelnya harus menghasilkan
    // gradien yang sama dengan versi Rust yang sudah diverifikasi beda hingga, termasuk
    // backward L2-normalisasi dan backward sigmoid yang ikut di dalamnya.
    {
        use asca_rs::delta::{DeltaBlock, DeltaCfg};
        let (batch, seq, d, h) = (8usize, 128usize, 192usize, 6usize);
        let t_len = batch * seq;
        let cfg = || DeltaCfg {
            d_model: d,
            n_heads: h,
        };
        // dua blok dari seed yang sama = bobot identik, satu dijalankan CPU satu di GPU
        let mut rng_a = Rng::new(21);
        let mut blk_cpu = DeltaBlock::new(cfg(), &mut rng_a);
        let mut rng_b = Rng::new(21);
        let mut blk_gpu = DeltaBlock::new(cfg(), &mut rng_b);
        let mut x = Mat::zeros(t_len, d);
        x.fill_rand(&mut rng, 0.5);
        let mut dy = Mat::zeros(t_len, d);
        dy.fill_rand(&mut rng, 0.1);

        cuda::set_enabled(false);
        let (_, c_cpu) = blk_cpu.forward(&x, batch);
        let dx_cpu = blk_cpu.backward(&c_cpu, &dy);
        cuda::set_enabled(true);
        let (_, c_gpu) = blk_gpu.forward(&x, batch);
        if c_gpu.dev.is_none() {
            println!("  delta blok: jalur GPU tidak aktif (ambang/ASCA_GPU_OFF?) - dilewati");
        } else {
            let dx_gpu = blk_gpu.backward(&c_gpu, &dy);
            let mut worst_rel = 0.0f32;
            let cmp = |name: &str, a: &Mat, b: &Mat, worst: &mut f32| {
                let mut e = 0.0f32;
                let mut s = 1e-9f32;
                for i in 0..a.d.len() {
                    e = e.max((a.d[i] - b.d[i]).abs());
                    s = s.max(a.d[i].abs());
                }
                *worst = worst.max(e / s);
                println!("    {name:<14} galat {e:.3e} (relatif {:.3e})", e / s);
            };
            cmp("dx", &dx_cpu, &dx_gpu, &mut worst_rel);
            let pc = blk_cpu.params();
            let pg = blk_gpu.params();
            for (a, b) in pc.iter().zip(pg.iter()) {
                cmp(&a.name, &a.grad, &b.grad, &mut worst_rel);
            }
            println!("  delta blok gradien: galat relatif terburuk {worst_rel:.3e}");
        }
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

    assert!(
        !arg(args, "--data", "").is_empty() || !arg(args, "--tokenized", "").is_empty(),
        "butuh --data FILE atau --tokenized FILE"
    );
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
