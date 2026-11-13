//! Harness perbandingan ASCA v2 vs baseline dense **di engine yang sama**.
//!
//! Ini inti dari metodologi: kedua model memakai `Mat`, `Param`, `Rng`, `RMSNorm`,
//! `AdamW`, dan loop training yang identik. Yang berbeda hanya arsitekturnya. Tanpa
//! ini, angka efisiensi hanya mengukur framework (kesalahan v1).

use crate::dense::DenseModel;
use crate::model::Model;
use crate::tensor::{AdamW, Param, Rng};
use crate::train::{Data, Split};

/// Antarmuka minimal supaya loop training bisa dipakai kedua arsitektur.
pub trait Trainable {
    fn forward_backward(&mut self, ids: &[u32], tgts: &[u32], batch: usize) -> f32;
    fn zero_grads(&mut self);
    fn params_mut(&mut self) -> Vec<&mut Param>;
    fn num_params(&self) -> usize;
    /// Jumlah float yang harus disimpan saat decode pada konteks `seq_len`.
    fn decode_state_floats(&self, seq_len: usize) -> usize;
    /// Jumlah parameter yang benar-benar dipakai per token.
    fn active_params(&self) -> usize;
}

impl Trainable for Model {
    fn forward_backward(&mut self, ids: &[u32], tgts: &[u32], batch: usize) -> f32 {
        let (logits, cache) = self.forward(ids, batch);
        self.backward(&cache, &logits, tgts)
    }
    fn zero_grads(&mut self) {
        Model::zero_grads(self)
    }
    fn params_mut(&mut self) -> Vec<&mut Param> {
        Model::params_mut(self)
    }
    fn num_params(&self) -> usize {
        Model::num_params(self)
    }
    fn decode_state_floats(&self, _seq_len: usize) -> usize {
        // state per head konstan: L * H * dh^2
        let dh = self.cfg.d_model / self.cfg.n_heads;
        self.cfg.n_layers * self.cfg.n_heads * dh * dh
    }
    fn active_params(&self) -> usize {
        // hash-MoE top-1: satu expert per token, jadi expert lain tidak terpakai
        let total = Model::num_params(self);
        let mut idle = 0usize;
        for l in self.layers.iter() {
            let per_expert = l.moe.w1.value.numel() / l.moe.n_experts
                + l.moe.w2.value.numel() / l.moe.n_experts
                + l.moe.d_ff
                + l.moe.b2.value.numel();
            idle += per_expert * (l.moe.n_experts - 1);
        }
        total - idle
    }
}

impl Trainable for DenseModel {
    fn forward_backward(&mut self, ids: &[u32], tgts: &[u32], batch: usize) -> f32 {
        let (logits, cache) = self.forward(ids, batch);
        self.backward(&cache, &logits, tgts)
    }
    fn zero_grads(&mut self) {
        DenseModel::zero_grads(self)
    }
    fn params_mut(&mut self) -> Vec<&mut Param> {
        DenseModel::params_mut(self)
    }
    fn num_params(&self) -> usize {
        DenseModel::num_params(self)
    }
    fn decode_state_floats(&self, seq_len: usize) -> usize {
        // KV cache tumbuh linear: L * 2 * T * d
        self.cfg.n_layers * 2 * seq_len * self.cfg.d_model
    }
    fn active_params(&self) -> usize {
        DenseModel::num_params(self)
    }
}

#[derive(Clone)]
pub struct BenchCfg {
    pub steps: usize,
    pub seq_len: usize,
    pub batch: usize,
    pub lr: f32,
    pub warmup: usize,
    pub eval_batches: usize,
}

pub struct BenchResult {
    pub label: String,
    pub params: usize,
    pub active_params: usize,
    pub decode_floats: usize,
    pub val_loss: f32,
    pub tok_per_sec: f32,
    pub seconds: f32,
}

fn lr_at(step: usize, cfg: &BenchCfg) -> f32 {
    if step < cfg.warmup {
        cfg.lr * (step as f32 + 1.0) / cfg.warmup as f32
    } else {
        let progress =
            (step - cfg.warmup) as f32 / (cfg.steps.max(cfg.warmup + 1) - cfg.warmup) as f32;
        let min_lr = cfg.lr * 0.1;
        min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

pub fn run<T: Trainable>(
    label: &str,
    model: &mut T,
    data: &Data,
    cfg: &BenchCfg,
    seed: u64,
) -> BenchResult {
    run_curve(label, model, data, cfg, seed, 0).0
}

/// Seperti [`run`], tetapi juga mencatat `(token_terlihat, val_loss)` tiap
/// `eval_every_steps` langkah. Dipakai untuk kurva loss vs token: satu-satunya cara
/// tahu apakah keunggulan bertahan atau hanya efek awal training.
pub fn run_curve<T: Trainable>(
    label: &str,
    model: &mut T,
    data: &Data,
    cfg: &BenchCfg,
    seed: u64,
    eval_every_steps: usize,
) -> (BenchResult, Vec<(usize, f32)>) {
    let mut curve: Vec<(usize, f32)> = Vec::new();
    let mut rng = Rng::new(seed);
    let mut opt = AdamW::new(cfg.lr);
    let t0 = std::time::Instant::now();
    // RNG evaluasi **tidak** bergantung pada seed training. Kalau bergantung, setiap seed
    // dievaluasi pada batch validasi yang berbeda, sehingga noise evaluasi ikut masuk ke
    // sebaran antar-seed dan membuat efek kecil (mis. delta -0.07) tidak bisa dipisahkan
    // dari noise. Dengan RNG tetap, semua seed diuji pada data yang sama dan delta
    // berpasangan menjadi jauh lebih tajam.
    let mut eval_rng = Rng::new(0xE7A1_0001);
    for step_i in 0..cfg.steps {
        opt.lr = lr_at(step_i, cfg);
        model.zero_grads();
        let (ids, tgts) = data.batch_seqs(&mut rng, cfg.seq_len, cfg.batch, Split::Train);
        model.forward_backward(&ids, &tgts, cfg.batch);
        for p in model.params_mut() {
            opt.update(p);
        }
        if eval_every_steps > 0 && (step_i + 1) % eval_every_steps == 0 {
            let mut vl = 0.0f32;
            for _ in 0..cfg.eval_batches {
                let (ids, tgts) = data.batch(&mut eval_rng, cfg.seq_len, Split::Val);
                model.zero_grads();
                vl += model.forward_backward(&ids, &tgts, 1);
            }
            model.zero_grads();
            let seen = (step_i + 1) * cfg.batch * cfg.seq_len;
            curve.push((seen, vl / cfg.eval_batches as f32));
        }
    }
    let seconds = t0.elapsed().as_secs_f32();
    let tokens = (cfg.steps * cfg.batch * cfg.seq_len) as f32;

    // eval
    let mut vl = 0.0f32;
    for _ in 0..cfg.eval_batches {
        let (ids, tgts) = data.batch(&mut rng, cfg.seq_len, Split::Val);
        model.zero_grads();
        vl += model.forward_backward(&ids, &tgts, 1);
    }
    model.zero_grads();
    vl /= cfg.eval_batches as f32;

    let res = BenchResult {
        label: label.to_string(),
        params: model.num_params(),
        active_params: model.active_params(),
        decode_floats: model.decode_state_floats(cfg.seq_len),
        val_loss: vl,
        tok_per_sec: tokens / seconds.max(1e-6),
        seconds,
    };
    (res, curve)
}

pub fn mean_std(xs: &[f32]) -> (f32, f32) {
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    if xs.len() < 2 {
        return (mean, 0.0);
    }
    let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n - 1.0);
    (mean, var.sqrt())
}
