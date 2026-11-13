//! Data, training loop, evaluasi, dan checkpoint biner (save/load).

use std::fs;
use std::io::{Read, Write};

use crate::model::{Config, Model, step};
use crate::tensor::{AdamW, Rng};

pub struct Data {
    pub bytes: Vec<u8>,
    pub train_end: usize,
}

impl Data {
    pub fn load(path: &str, val_frac: f32) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() < 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file terlalu kecil",
            ));
        }
        let train_end = ((bytes.len() as f32) * (1.0 - val_frac)) as usize;
        Ok(Self { bytes, train_end })
    }

    /// Ambil satu sekuens (ids, targets) mulai dari offset acak di rentang tertentu.
    pub fn batch(&self, rng: &mut Rng, seq_len: usize, split: Split) -> (Vec<u32>, Vec<u32>) {
        let (lo, hi) = match split {
            Split::Train => (0, self.train_end),
            Split::Val => (self.train_end, self.bytes.len()),
        };
        let span = hi.saturating_sub(lo).saturating_sub(seq_len + 1).max(1);
        let start = lo + rng.below(span);
        let ids = self.bytes[start..start + seq_len]
            .iter()
            .map(|&b| b as u32)
            .collect();
        let tgts = self.bytes[start + 1..start + 1 + seq_len]
            .iter()
            .map(|&b| b as u32)
            .collect();
        (ids, tgts)
    }
}

#[derive(Clone, Copy)]
pub enum Split {
    Train,
    Val,
}

pub struct TrainCfg {
    pub steps: usize,
    pub seq_len: usize,
    pub batch: usize,
    pub lr: f32,
    pub warmup: usize,
    pub eval_every: usize,
    pub eval_batches: usize,
    pub log_every: usize,
}

/// Satu langkah training: forward+backward `batch` sekuens, lalu update.
pub fn train_step(model: &mut Model, data: &Data, rng: &mut Rng, cfg: &TrainCfg) -> f32 {
    model.zero_grads();
    let mut loss = 0.0f32;
    for _ in 0..cfg.batch {
        let (ids, tgts) = data.batch(rng, cfg.seq_len, Split::Train);
        let (logits, cache) = model.forward(&ids);
        loss += model.backward(&cache, &logits, &tgts);
    }
    // rata-rata gradien antar sekuens
    let inv = 1.0 / cfg.batch as f32;
    for p in model.params_mut() {
        for g in p.grad.d.iter_mut() {
            *g *= inv;
        }
    }
    loss / cfg.batch as f32
}

pub fn evaluate(model: &mut Model, data: &Data, rng: &mut Rng, cfg: &TrainCfg) -> f32 {
    let mut total = 0.0f32;
    for _ in 0..cfg.eval_batches {
        let (ids, tgts) = data.batch(rng, cfg.seq_len, Split::Val);
        let (logits, cache) = model.forward(&ids);
        // backward hanya untuk menghitung loss (gradien dibuang)
        model.zero_grads();
        total += model.backward(&cache, &logits, &tgts);
    }
    model.zero_grads();
    total / cfg.eval_batches as f32
}

pub fn lr_at(step: usize, cfg: &TrainCfg) -> f32 {
    if step < cfg.warmup {
        cfg.lr * (step as f32 + 1.0) / cfg.warmup as f32
    } else {
        let progress =
            (step - cfg.warmup) as f32 / (cfg.steps.max(cfg.warmup + 1) - cfg.warmup) as f32;
        // cosine decay ke 10% lr
        let min_lr = cfg.lr * 0.1;
        min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

pub fn train(model: &mut Model, data: &Data, cfg: &TrainCfg, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut opt = AdamW::new(cfg.lr);
    println!(
        "params: {}  seq_len: {}  batch: {}  steps: {}",
        model.num_params(),
        cfg.seq_len,
        cfg.batch,
        cfg.steps
    );
    let t0 = std::time::Instant::now();
    for step_i in 0..cfg.steps {
        opt.lr = lr_at(step_i, cfg);
        let loss = train_step(model, data, &mut rng, cfg);
        step(model, &mut opt);
        if cfg.log_every > 0 && (step_i + 1) % cfg.log_every == 0 {
            let el = t0.elapsed().as_secs_f32();
            let tok = ((step_i + 1) * cfg.batch * cfg.seq_len) as f32;
            println!(
                "step {:5}/{}  loss {:.4}  lr {:.2e}  {:.0} tok/s  {:.1}s",
                step_i + 1,
                cfg.steps,
                loss,
                opt.lr,
                tok / el.max(1e-6),
                el
            );
        }
        if cfg.eval_every > 0 && (step_i + 1) % cfg.eval_every == 0 {
            let vl = evaluate(model, data, &mut rng, cfg);
            println!(
                "  eval step {}: val_loss {:.4}  ppl {:.2}",
                step_i + 1,
                vl,
                vl.exp()
            );
        }
    }
}

// ------------------------------------------------------------------ checkpoint
const MAGIC: &[u8; 8] = b"ASCA2CK1";

pub fn save(path: &str, model: &Model) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(MAGIC)?;
    let c = &model.cfg;
    for v in [
        c.vocab,
        c.d_model,
        c.n_heads,
        c.n_layers,
        c.n_experts,
        c.d_ff,
    ] {
        f.write_all(&(v as u64).to_le_bytes())?;
    }
    for p in model.params() {
        f.write_all(&(p.numel() as u64).to_le_bytes())?;
        for v in p.value.d.iter() {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn load(path: &str, seq_hint: usize) -> std::io::Result<(Model, Config)> {
    let mut f = fs::File::open(path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bukan checkpoint ASCA v2",
        ));
    }
    let mut nums = [0u64; 6];
    for n in nums.iter_mut() {
        let mut b = [0u8; 8];
        f.read_exact(&mut b)?;
        *n = u64::from_le_bytes(b);
    }
    let cfg = Config {
        vocab: nums[0] as usize,
        d_model: nums[1] as usize,
        n_heads: nums[2] as usize,
        n_layers: nums[3] as usize,
        n_experts: nums[4] as usize,
        d_ff: nums[5] as usize,
    };
    let mut rng = Rng::new(1); // nilai awal tidak penting, akan ditimpa
    let mut model = Model::new(cfg.clone(), &mut rng);
    let n_params = model.params().len();
    for i in 0..n_params {
        let mut b = [0u8; 8];
        f.read_exact(&mut b)?;
        let n = u64::from_le_bytes(b) as usize;
        let mut buf = vec![0.0f32; n];
        for v in buf.iter_mut() {
            let mut fb = [0u8; 4];
            f.read_exact(&mut fb)?;
            *v = f32::from_le_bytes(fb);
        }
        let p = &mut model.params_mut()[i];
        if p.numel() != n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("ukuran param {i} tidak cocok: {n} vs {}", p.numel()),
            ));
        }
        p.value.d.copy_from_slice(&buf);
    }
    let _ = seq_hint;
    Ok((model, cfg))
}
