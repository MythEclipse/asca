//! Gate paritas decode: menjalankan model satu token demi satu dengan state yang dibawa
//! harus menghasilkan logits yang sama dengan satu forward penuh atas sekuens itu.
//!
//! Tanpa gate ini, "state decode konstan 2.048 float" hanya klaim di dokumen: jalur
//! decode bisa saja diam-diam membuang informasi (mis. state tidak diperbarui, atau
//! gate alpha/beta tertukar) dan tetap menghasilkan teks yang terlihat wajar.

use asca_rs::model::{Config, Model};
use asca_rs::tensor::Rng;

fn main() {
    let mut rng = Rng::new(3);
    let seq = 24usize;
    let ids: Vec<u32> = (0..seq).map(|_| rng.below(256) as u32).collect();
    let model = Model::new(Config::tiny(), &mut Rng::new(17));

    let (logits, _) = model.forward(&ids, 1);
    let mut st = model.new_decode_state();
    let mut worst = 0.0f32;
    let mut scale = 1e-9f32;
    for (t, &id) in ids.iter().enumerate() {
        let step_logits = model.step(id, &mut st);
        for c in 0..logits.cols {
            let a = logits.at(t, c);
            let b = step_logits.at(0, c);
            worst = worst.max((a - b).abs());
            scale = scale.max(a.abs());
        }
    }
    let rel = worst / scale;
    println!("paritas decode: galat absolut terburuk {worst:.3e}, relatif {rel:.3e}");
    if rel > 1e-4 {
        println!("PARITAS DECODE GAGAL");
        std::process::exit(1);
    }
    let floats = model
        .new_decode_state()
        .delta
        .iter()
        .flatten()
        .map(|v| v.len())
        .sum::<usize>();
    println!("state decode: {floats} float, konstan terhadap panjang konteks");
    println!("Paritas decode cocok.");
}
