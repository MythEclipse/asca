//! Verifikasi gradien backward tulisan tangan dengan beda hingga.

use asca_rs::delta::{DeltaBlock, DeltaCfg};
use asca_rs::tensor::{Mat, Rng};

fn loss_of(block: &DeltaBlock, x: &Mat, dy: &Mat) -> f32 {
    let (y, _) = block.forward(x);
    let mut l = 0.0f32;
    for r in 0..y.rows {
        for c in 0..y.cols {
            l += y.at(r, c) * dy.at(r, c);
        }
    }
    l
}

fn check_param(name: &str, block: &mut DeltaBlock, x: &Mat, dy: &Mat, fails: &mut usize) {
    // gradien analitik
    for p in block.params_mut() {
        p.zero_grad();
    }
    let (_, cache) = block.forward(x);
    let _dx = block.backward(&cache, dy);

    let eps = 1e-3f32;
    let mut checked = 0;
    let mut worst = 0.0f32;
    for idx in 0..block.params().len() {
        if block.params()[idx].name != name {
            continue;
        }
        let numel = block.params()[idx].numel();
        for i in (0..numel).step_by((numel / 6).max(1)) {
            let orig = block.params()[idx].value.d[i];
            let analytic = block.params()[idx].grad.d[i];

            block.params_mut()[idx].value.d[i] = orig + eps;
            let lp = loss_of(block, x, dy);
            block.params_mut()[idx].value.d[i] = orig - eps;
            let lm = loss_of(block, x, dy);
            block.params_mut()[idx].value.d[i] = orig;

            let numeric = (lp - lm) / (2.0 * eps);
            // f32 + beda hingga: derau absolut ~1e-4. Kriteria gabungan relatif
            // terhadap nilai DAN terhadap skala global gradien, supaya entri kecil
            // tidak gagal palsu sementara bug nyata (O(0.1)) tetap tertangkap.
            let scale = block.params()[idx].grad.max_abs().max(1e-3);
            let tol = 2e-2 * analytic.abs().max(numeric.abs()) + 1e-3 * scale;
            let rel = (analytic - numeric).abs() / tol;
            worst = worst.max(rel);
            checked += 1;
            if rel > 1.0 {
                println!(
                    "  GAGAL {name}[{i}]: analitik {analytic:.6e} vs numerik {numeric:.6e} (rel {rel:.2e})"
                );
                *fails += 1;
            }
        }
    }
    println!("  {name}: {checked} entri diperiksa, galat relatif terburuk {worst:.2e}");
}

fn main() {
    let mut rng = Rng::new(7);
    let cfg = DeltaCfg {
        d_model: 16,
        n_heads: 2,
    };
    let mut block = DeltaBlock::new(cfg, &mut rng);
    let t_len = 12;
    let mut x = Mat::zeros(t_len, 16);
    x.fill_rand(&mut rng, 1.0);
    let mut dy = Mat::zeros(t_len, 16);
    dy.fill_rand(&mut rng, 1.0);

    let mut fails = 0;
    println!("== gradcheck parameter delta block");
    for name in [
        "delta.w_qkv",
        "delta.b_qkv",
        "delta.w_gate",
        "delta.b_gate",
        "delta.w_o",
        "delta.b_o",
    ] {
        check_param(name, &mut block, &x, &dy, &mut fails);
    }

    // gradcheck dx
    for p in block.params_mut() {
        p.zero_grad();
    }
    let (_, cache) = block.forward(&x);
    let dx = block.backward(&cache, &dy);
    let eps = 1e-3f32;
    let mut worst = 0.0f32;
    for i in 0..x.d.len() {
        let orig = x.d[i];
        let mut xp = x.clone();
        xp.d[i] = orig + eps;
        let lp = loss_of(&block, &xp, &dy);
        xp.d[i] = orig - eps;
        let lm = loss_of(&block, &xp, &dy);
        let numeric = (lp - lm) / (2.0 * eps);
        let scale = dx.max_abs().max(1e-3);
        let tol = 2e-2 * dx.d[i].abs().max(numeric.abs()) + 1e-3 * scale;
        let rel = (dx.d[i] - numeric).abs() / tol;
        worst = worst.max(rel);
        if rel > 1.0 {
            println!(
                "  GAGAL dx[{i}]: analitik {:.6e} vs numerik {numeric:.6e}",
                dx.d[i]
            );
            fails += 1;
        }
    }
    println!(
        "  dx: {} entri, galat relatif terburuk {worst:.2e}",
        x.d.len()
    );

    if fails > 0 {
        println!("\n{fails} entri GAGAL");
        std::process::exit(1);
    }
    println!("\nSemua gradien cocok dengan beda hingga.");
}
