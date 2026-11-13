//! Gradcheck baseline dense: backward attention + softmax + RoPE + FFN.
//!
//! Kriteria sadar-derau yang sama dengan `gradcheck_model.rs`: entri bergradien
//! terbesar, eps=3e-4, entri di bawah 3x derau f32 dilewati.

use asca_rs::dense::{DenseCfg, DenseModel};
use asca_rs::tensor::Rng;

fn main() {
    let mut rng = Rng::new(21);
    let cfg = DenseCfg {
        vocab: 256,
        d_model: 16,
        n_heads: 2,
        n_layers: 1,
        d_ff: 24,
    };
    let mut model = DenseModel::new(cfg, &mut rng);
    let ids: Vec<u32> = (0..10).map(|_| rng.below(256) as u32).collect();
    let tgts: Vec<u32> = (0..10).map(|_| rng.below(256) as u32).collect();

    model.zero_grads();
    let (logits, cache) = model.forward(&ids);
    let loss = model.backward(&cache, &logits, &tgts);
    println!("loss awal {loss:.6}");

    let eps = 3e-4f32;
    let ulp = loss.abs().max(1.0) * f32::EPSILON;
    let noise = 4.0 * ulp / (2.0 * eps);
    let mut fails = 0;
    let mut worst = 0.0f32;
    let names: Vec<String> = model.params().iter().map(|p| p.name.clone()).collect();
    for name in &names {
        let idx = names.iter().position(|n| n == name).unwrap();
        let numel = model.params()[idx].numel();
        let mut order: Vec<usize> = (0..numel).collect();
        let g = &model.params()[idx].grad.d;
        order.sort_by(|&a, &b| g[b].abs().partial_cmp(&g[a].abs()).unwrap());
        let mut checked = 0;
        let mut skipped = 0;
        for &i in order.iter().take(4) {
            let orig = model.params()[idx].value.d[i];
            let analytic = model.params()[idx].grad.d[i];
            if analytic.abs() < 3.0 * noise {
                skipped += 1;
                continue;
            }
            model.params_mut()[idx].value.d[i] = orig + eps;
            let lp = model.loss(&ids, &tgts);
            model.params_mut()[idx].value.d[i] = orig - eps;
            let lm = model.loss(&ids, &tgts);
            model.params_mut()[idx].value.d[i] = orig;
            let numeric = (lp - lm) / (2.0 * eps);
            let tol = 0.05 * analytic.abs() + noise;
            let rel = (analytic - numeric).abs() / tol;
            worst = worst.max(rel);
            checked += 1;
            if rel > 1.0 {
                println!("  GAGAL {name}[{i}]: analitik {analytic:.5e} vs numerik {numeric:.5e}");
                fails += 1;
            }
        }
        println!("  {name}: {checked} diperiksa, {skipped} di bawah derau");
    }
    println!("galat relatif terburuk (terhadap toleransi): {worst:.3e}");
    if fails > 0 {
        println!("\n{fails} entri GAGAL");
        std::process::exit(1);
    }
    println!("\nSemua gradien baseline dense cocok dengan beda hingga.");
}
