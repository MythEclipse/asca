//! Gate paritas batching: satu forward/backward dengan `batch=4` harus identik dengan
//! empat forward/backward `batch=1` yang dirata-ratakan. Tanpa gate ini, batching bisa
//! diam-diam membocorkan token antar sekuens (misalnya attention atau rekurensi yang
//! tidak di-reset di batas sekuens) dan hasil training tetap terlihat wajar.

use asca_rs::dense::{DenseCfg, DenseModel};
use asca_rs::model::{Config, Model};
use asca_rs::tensor::Rng;

fn cmp(label: &str, a: f32, b: f32, tol: f32) -> bool {
    let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-6);
    let ok = rel <= tol;
    println!(
        "  {label}: batch=1 {a:.6} vs batch=B {b:.6} (rel {rel:.2e}) {}",
        if ok { "OK" } else { "GAGAL" }
    );
    ok
}

fn main() {
    let mut rng = Rng::new(5);
    let batch = 4usize;
    let seq = 12usize;
    let ids: Vec<u32> = (0..batch * seq).map(|_| rng.below(256) as u32).collect();
    let tgts: Vec<u32> = (0..batch * seq).map(|_| rng.below(256) as u32).collect();
    let mut ok = true;

    // ---- ASCA
    println!("ASCA v2:");
    let cfg = Config::tiny();
    let mut a1 = Model::new(cfg.clone(), &mut Rng::new(9));
    let mut ab = Model::new(cfg, &mut Rng::new(9));
    let mut loss1 = 0.0f32;
    a1.zero_grads();
    for b in 0..batch {
        let s = b * seq..(b + 1) * seq;
        let (logits, cache) = a1.forward(&ids[s.clone()], 1);
        loss1 += a1.backward(&cache, &logits, &tgts[s]);
    }
    loss1 /= batch as f32;
    for p in a1.params_mut() {
        for g in p.grad.d.iter_mut() {
            *g /= batch as f32;
        }
    }
    ab.zero_grads();
    let (logits, cache) = ab.forward(&ids, batch);
    let lossb = ab.backward(&cache, &logits, &tgts);
    ok &= cmp("loss", loss1, lossb, 1e-5);

    let n1: Vec<String> = a1.params().iter().map(|p| p.name.clone()).collect();
    let mut worst = 0.0f32;
    for (i, name) in n1.iter().enumerate() {
        let g1 = &a1.params()[i].grad.d;
        let gb = &ab.params()[i].grad.d;
        let mut num = 0.0f32;
        let mut den = 1e-9f32;
        for k in 0..g1.len() {
            num = num.max((g1[k] - gb[k]).abs());
            den = den.max(g1[k].abs());
        }
        let rel = num / den;
        worst = worst.max(rel);
        if rel > 2e-4 {
            println!("  grad {name}: rel {rel:.2e} GAGAL");
            ok = false;
        }
    }
    println!("  gradien: galat relatif terburuk {worst:.2e}");

    // ---- dense
    println!("dense:");
    let dcfg = DenseCfg {
        vocab: 256,
        d_model: 16,
        n_heads: 2,
        n_layers: 1,
        d_ff: 24,
    };
    let mut d1 = DenseModel::new(
        DenseCfg {
            vocab: dcfg.vocab,
            d_model: dcfg.d_model,
            n_heads: dcfg.n_heads,
            n_layers: dcfg.n_layers,
            d_ff: dcfg.d_ff,
        },
        &mut Rng::new(11),
    );
    let mut db = DenseModel::new(dcfg, &mut Rng::new(11));
    let mut dl1 = 0.0f32;
    d1.zero_grads();
    for b in 0..batch {
        let s = b * seq..(b + 1) * seq;
        let (logits, cache) = d1.forward(&ids[s.clone()], 1);
        dl1 += d1.backward(&cache, &logits, &tgts[s]);
    }
    dl1 /= batch as f32;
    for p in d1.params_mut() {
        for g in p.grad.d.iter_mut() {
            *g /= batch as f32;
        }
    }
    db.zero_grads();
    let (logits, cache) = db.forward(&ids, batch);
    let dlb = db.backward(&cache, &logits, &tgts);
    ok &= cmp("loss", dl1, dlb, 1e-5);
    let n2: Vec<String> = d1.params().iter().map(|p| p.name.clone()).collect();
    let mut worst2 = 0.0f32;
    for (i, name) in n2.iter().enumerate() {
        let g1 = &d1.params()[i].grad.d;
        let gb = &db.params()[i].grad.d;
        let mut num = 0.0f32;
        let mut den = 1e-9f32;
        for k in 0..g1.len() {
            num = num.max((g1[k] - gb[k]).abs());
            den = den.max(g1[k].abs());
        }
        let rel = num / den;
        worst2 = worst2.max(rel);
        if rel > 2e-4 {
            println!("  grad {name}: rel {rel:.2e} GAGAL");
            ok = false;
        }
    }
    println!("  gradien: galat relatif terburuk {worst2:.2e}");

    if !ok {
        println!("\nPARITAS BATCH GAGAL");
        std::process::exit(1);
    }
    println!("\nParitas batch cocok untuk kedua arsitektur.");
}
