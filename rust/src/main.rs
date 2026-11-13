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
        _ => {
            println!("asca — ASCA v2 (tanpa attention, state konstan, Rust murni)");
            println!("pakai:");
            println!("  asca train   --data FILE [--steps 2000] [--seq-len 128] [--batch 8] [--lr 3e-3]");
            println!("               [--d-model 64] [--n-heads 4] [--n-layers 2] [--n-experts 8]");
            println!("               [--d-ff 128] [--out ckpt.bin] [--eval-every 200]");
            println!("  asca eval    --ckpt FILE --data FILE [--batches 20] [--seq-len 128]");
            println!("  asca sample  --ckpt FILE [--prompt TEXT] [--tokens 200]");
            println!("  asca inspect --ckpt FILE");
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
    train::train(&mut model, &data, &tcfg, arg(args, "--seed", "0").parse().unwrap());
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
        println!("  {:<16} {:>8} x {:<8} = {}", p.name, p.value.rows, p.value.cols, p.numel());
    }
}
