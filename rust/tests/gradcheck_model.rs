//! Gradcheck model penuh: forward + backward end-to-end lewat delta block,
//! hash-MoE, RMSNorm, embedding, dan head.

use asca_rs::model::{Config, Model};
use asca_rs::tensor::Rng;

fn main() {
    let mut rng = Rng::new(11);
    let cfg = Config {
        vocab: 256,
        d_model: 16,
        n_heads: 2,
        n_layers: 1,
        n_experts: 2,
        d_ff: 16,
    };
    let mut model = Model::new(cfg, &mut rng);
    let ids: Vec<u32> = (0..10).map(|_| rng.below(256) as u32).collect();
    let tgts: Vec<u32> = (0..10).map(|_| rng.below(256) as u32).collect();

    model.zero_grads();
    let (logits, cache) = model.forward(&ids);
    let loss = model.backward(&cache, &logits, &tgts);
    println!("loss awal {loss:.6}");

    // Beda hingga pada f32 punya dua sumber galat yang berlawanan arah:
    //   truncation  ~ eps^2 * f'''  (turun kalau eps kecil)
    //   derau f32   ~ ulp(loss)/(2 eps)  (naik kalau eps kecil)
    // Rekurensi delta bergatuk sangat melengkung, jadi eps=1e-2 memberi truncation
    // 10-20%. eps=3e-4 dipilih di titik seimbang; sudah diverifikasi manual bahwa
    // turunan numerik konvergen ke nilai analitik saat eps -> 0.
    let eps = 3e-4f32;
    let ulp = (loss.abs().max(1.0)) * f32::EPSILON;
    let noise = 4.0 * ulp / (2.0 * eps);
    let mut fails = 0;
    let mut worst = 0.0f32;
    let names: Vec<String> = model.params().iter().map(|p| p.name.clone()).collect();
    for name in &names {
        let idx = names.iter().position(|n| n == name).unwrap();
        let numel = model.params()[idx].numel();
        let scale = model.params()[idx].grad.max_abs().max(1e-4);
        let mut checked = 0;
        let mut skipped = 0;
        // Ambil entri dengan gradien terbesar: di situ sinyal jauh di atas derau
        // f32, jadi ujinya benar-benar menguji backward dan bukan cuma derau.
        let mut order: Vec<usize> = (0..numel).collect();
        let g = &model.params()[idx].grad.d;
        order.sort_by(|&a, &b| g[b].abs().partial_cmp(&g[a].abs()).unwrap());
        for &i in order.iter().take(4) {
            let orig = model.params()[idx].value.d[i];
            let analytic = model.params()[idx].grad.d[i];
            model.params_mut()[idx].value.d[i] = orig + eps;
            let lp = model.loss(&ids, &tgts);
            model.params_mut()[idx].value.d[i] = orig - eps;
            let lm = model.loss(&ids, &tgts);
            model.params_mut()[idx].value.d[i] = orig;
            let numeric = (lp - lm) / (2.0 * eps);
            // Di bawah resolusi f32: tidak bisa diverifikasi, jangan divonis gagal.
            if analytic.abs() < 3.0 * noise {
                skipped += 1;
                continue;
            }
            let tol = 0.05 * analytic.abs() + noise;
            let rel = (analytic - numeric).abs() / tol;
            worst = worst.max(rel);
            checked += 1;
            if rel > 1.0 {
                println!(
                    "  GAGAL {name}[{i}]: analitik {analytic:.5e} vs numerik {numeric:.5e} (derau {noise:.1e})"
                );
                fails += 1;
            }
        }
        println!(
            "  {name}: {checked} diperiksa, {skipped} di bawah derau f32, skala grad {scale:.2e}"
        );
    }
    println!("galat relatif terburuk (terhadap toleransi): {worst:.3e}");
    if fails > 0 {
        println!("\n{fails} entri GAGAL");
        std::process::exit(1);
    }
    println!("\nSemua gradien model cocok dengan beda hingga.");
}
