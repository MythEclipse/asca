//! Data, training loop, evaluasi, dan checkpoint biner (save/load).

use std::fs;
use std::io::{Read, Write};

use crate::model::{Config, Model, step};
use crate::tensor::{AdamW, Rng};

pub struct Data {
    /// id token (byte sebagai u32 kalau tanpa tokenizer, subword kalau dengan BPE)
    pub tokens: Vec<u32>,
    pub train_end: usize,
    /// jumlah byte asli, untuk pelaporan kompresi
    pub raw_bytes: usize,
}

impl Data {
    /// Muat data dengan tokenizer BPE. Model melihat subword, bukan byte.
    pub fn load_with_bpe(
        path: &str,
        val_frac: f32,
        bpe: &crate::bpe::Bpe,
    ) -> std::io::Result<Self> {
        let raw = fs::read(path)?;
        if raw.len() < 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file terlalu kecil",
            ));
        }
        let tokens = bpe.encode(&raw);
        let train_end = ((tokens.len() as f32) * (1.0 - val_frac)) as usize;
        Ok(Self {
            tokens,
            train_end,
            raw_bytes: raw.len(),
        })
    }

    /// Simpan id token yang sudah di-encode, supaya proses berikutnya tidak perlu
    /// meng-encode ulang korpus (26 detik untuk 100 MB, dan itu dibayar setiap kali
    /// menjalankan sweep LR atau melanjutkan training).
    /// Bits per byte: ukuran kualitas yang TIDAK bergantung tokenizer.
    ///
    /// `val_loss` adalah nats per token, jadi model dengan vocab berbeda tidak bisa
    /// dibandingkan langsung - vocab besar otomatis punya loss per token lebih tinggi
    /// walau kemampuannya sama. Membagi dengan byte/token dan ln(2) menghasilkan bits
    /// per byte, satuan yang sama dipakai paper bahasa lain sehingga angkanya bisa
    /// diletakkan berdampingan dengan hasil orang lain.
    pub fn bits_per_byte(&self, val_loss: f32) -> f32 {
        let bpt = self.raw_bytes as f32 / self.tokens.len() as f32;
        val_loss / std::f32::consts::LN_2 / bpt
    }

    pub fn save_tokens(&self, path: &str, vocab: usize) -> std::io::Result<()> {
        let mut f = fs::File::create(path)?;
        f.write_all(b"ASCATOK1")?;
        f.write_all(&(self.tokens.len() as u64).to_le_bytes())?;
        f.write_all(&(self.train_end as u64).to_le_bytes())?;
        f.write_all(&(self.raw_bytes as u64).to_le_bytes())?;
        // ukuran vocab ikut disimpan: tanpa ini model akan dibangun dengan vocab 256
        // padahal tokennya sampai 12287, dan embedding-nya salah ukuran.
        f.write_all(&(vocab as u64).to_le_bytes())?;
        let mut buf = Vec::with_capacity(self.tokens.len() * 4);
        for t in self.tokens.iter() {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        f.write_all(&buf)
    }

    /// Ukuran vocab dari header file token, tanpa memuat seluruh tokennya.
    pub fn vocab_of_tokens(path: &str) -> std::io::Result<usize> {
        let buf = fs::read(path)?;
        if buf.len() < 40 || &buf[..8] != b"ASCATOK1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bukan file token ASCA",
            ));
        }
        Ok(u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize)
    }

    pub fn load_tokens(path: &str, val_frac: f32) -> std::io::Result<Self> {
        let buf = fs::read(path)?;
        if buf.len() < 40 || &buf[..8] != b"ASCATOK1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bukan file token ASCA",
            ));
        }
        let n = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let _train_end_saved = u64::from_le_bytes(buf[16..24].try_into().unwrap()) as usize;
        let raw_bytes = u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize;
        let mut tokens = Vec::with_capacity(n);
        let mut off = 40;
        for _ in 0..n {
            tokens.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        let train_end = ((tokens.len() as f32) * (1.0 - val_frac)) as usize;
        Ok(Self {
            tokens,
            train_end,
            raw_bytes,
        })
    }

    pub fn load(path: &str, val_frac: f32) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() < 1024 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file terlalu kecil",
            ));
        }
        let train_end = ((bytes.len() as f32) * (1.0 - val_frac)) as usize;
        let raw_bytes = bytes.len();
        Ok(Self {
            tokens: bytes.iter().map(|&b| b as u32).collect(),
            train_end,
            raw_bytes,
        })
    }

    /// Ambil satu sekuens (ids, targets) mulai dari offset acak di rentang tertentu.
    /// `batch` sekuens ditumpuk menjadi satu vektor `B*seq_len`, supaya satu langkah
    /// cukup satu forward/backward dan semua GEMM memakai `B*seq_len` baris.
    pub fn batch_seqs(
        &self,
        rng: &mut Rng,
        seq_len: usize,
        batch: usize,
        split: Split,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut ids = Vec::with_capacity(seq_len * batch);
        let mut tgts = Vec::with_capacity(seq_len * batch);
        for _ in 0..batch {
            let (a, b) = self.batch(rng, seq_len, split);
            ids.extend_from_slice(&a);
            tgts.extend_from_slice(&b);
        }
        (ids, tgts)
    }

    pub fn batch(&self, rng: &mut Rng, seq_len: usize, split: Split) -> (Vec<u32>, Vec<u32>) {
        let (lo, hi) = match split {
            Split::Train => (0, self.train_end),
            Split::Val => (self.train_end, self.tokens.len()),
        };
        let span = hi.saturating_sub(lo).saturating_sub(seq_len + 1).max(1);
        let start = lo + rng.below(span);
        let ids = self.tokens[start..start + seq_len].to_vec();
        let tgts = self.tokens[start + 1..start + 1 + seq_len].to_vec();
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
    crate::prof::time(15, || model.zero_grads());
    let (ids, tgts) = crate::prof::time(16, || {
        data.batch_seqs(rng, cfg.seq_len, cfg.batch, Split::Train)
    });
    let (logits, cache) = crate::prof::time(17, || model.forward(&ids, cfg.batch));
    let loss = crate::prof::time(18, || model.backward(&cache, &logits, &tgts));
    // rata-rata gradien antar sekuens
    // loss dan gradien sudah dirata-ratakan per token di dalam backward
    loss
}

pub fn evaluate(model: &mut Model, data: &Data, rng: &mut Rng, cfg: &TrainCfg) -> f32 {
    let mut total = 0.0f32;
    for _ in 0..cfg.eval_batches {
        let (ids, tgts) = data.batch(rng, cfg.seq_len, Split::Val);
        let (logits, cache) = model.forward(&ids, 1);
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
    train_resume(model, data, cfg, seed, AdamW::new(cfg.lr), 0, |_, _, _| {
        Ok(())
    })
    .expect("gagal simpan checkpoint");
}

/// Training yang bisa dilanjutkan: mulai dari `start_step`, memakai optimizer yang sudah
/// berisi momen, dan memanggil `on_save` secara berkala.
pub fn train_resume<F>(
    model: &mut Model,
    data: &Data,
    cfg: &TrainCfg,
    seed: u64,
    mut opt: AdamW,
    start_step: usize,
    mut on_save: F,
) -> std::io::Result<()>
where
    F: FnMut(&Model, &AdamW, usize) -> std::io::Result<()>,
{
    let mut rng = Rng::new(seed);
    println!(
        "params: {}  seq_len: {}  batch: {}  steps: {}",
        model.num_params(),
        cfg.seq_len,
        cfg.batch,
        cfg.steps
    );
    let t0 = std::time::Instant::now();
    for step_i in start_step..cfg.steps {
        opt.lr = lr_at(step_i, cfg);
        let loss = train_step(model, data, &mut rng, cfg);
        crate::prof::time(14, || step(model, &mut opt));
        if cfg.log_every > 0 && (step_i + 1) % cfg.log_every == 0 {
            let el = t0.elapsed().as_secs_f32();
            // Setelah resume, `step_i` dimulai dari start_step. Menghitung token dari 0
            // membuat tok/s terlihat jauh lebih tinggi dari kenyataan (bug yang sempat
            // saya buat: 250 langkah dilaporkan sebagai 2.750).
            let tok = ((step_i + 1 - start_step) * cfg.batch * cfg.seq_len) as f32;
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
                "  eval step {}: val_loss {:.4}  ppl {:.2}  bpb {:.4}",
                step_i + 1,
                vl,
                vl.exp(),
                data.bits_per_byte(vl)
            );
            // simpan berkala supaya run panjang tidak hilang kalau terputus
            on_save(model, &opt, step_i + 1)?;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ checkpoint
const MAGIC: &[u8; 8] = b"ASCA2CK1";
/// Checkpoint yang bisa dilanjutkan: bobot + momen AdamW + nomor langkah.
const MAGIC_R: &[u8; 8] = b"ASCARSM1";

/// Simpan bobot, momen AdamW, dan nomor langkah supaya training panjang bisa dilanjutkan
/// persis, bukan diulang dari nol. Tanpa momen, lanjutan akan "kaget" dan loss melonjak.
pub fn save_resume(path: &str, model: &Model, step: usize) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(MAGIC_R)?;
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
    f.write_all(&(step as u64).to_le_bytes())?;
    for p in model.params() {
        f.write_all(&(p.numel() as u64).to_le_bytes())?;
        for m in [&p.value, &p.m, &p.v] {
            for v in m.d.iter() {
                f.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

pub fn load_resume(path: &str) -> std::io::Result<(Model, AdamW, usize)> {
    let buf = fs::read(path)?;
    if buf.len() < 64 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "checkpoint terlalu kecil",
        ));
    }
    let magic = &buf[..8];
    if magic != MAGIC_R {
        // format lama (bobot saja): bisa dimuat, tapi mulai dari langkah 0
        let (m, _c) = load(path, 0)?;
        return Ok((m, AdamW::new(3e-3), 0));
    }
    let mut off = 8;
    let mut nums = [0u64; 6];
    for n in nums.iter_mut() {
        *n = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        off += 8;
    }
    let step = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap()) as usize;
    off += 8;
    let cfg = Config {
        vocab: nums[0] as usize,
        d_model: nums[1] as usize,
        n_heads: nums[2] as usize,
        n_layers: nums[3] as usize,
        n_experts: nums[4] as usize,
        d_ff: nums[5] as usize,
    };
    let mut rng = Rng::new(1);
    let mut model = Model::new(cfg, &mut rng);
    let n_params = model.params().len();
    for i in 0..n_params {
        let n = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        let mut vals: Vec<Vec<f32>> = Vec::with_capacity(3);
        for _ in 0..3 {
            let mut v = vec![0.0f32; n];
            for x in v.iter_mut() {
                *x = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                off += 4;
            }
            vals.push(v);
        }
        let p = &mut model.params_mut()[i];
        if p.numel() != n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("ukuran param {i} tidak cocok: {n} vs {}", p.numel()),
            ));
        }
        p.value.d.copy_from_slice(&vals[0]);
        p.m.d.copy_from_slice(&vals[1]);
        p.v.d.copy_from_slice(&vals[2]);
    }
    let mut opt = AdamW::new(3e-3);
    opt.step = step as u32;
    Ok((model, opt, step))
}

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
    // Terima kedua format: checkpoint bobot-saja (MAGIC) dan checkpoint yang bisa
    // dilanjutkan (MAGIC_R, berisi momen AdamW). Tanpa ini, model yang disimpan oleh
    // training terbaru tidak bisa dipakai untuk sample/chat.
    {
        let mut head = [0u8; 8];
        use std::io::Read as _;
        if let Ok(mut fh) = fs::File::open(path) {
            if fh.read_exact(&mut head).is_ok() && &head == MAGIC_R {
                let (m, _opt, _step) = load_resume(path)?;
                let cfg = m.cfg.clone();
                return Ok((m, cfg));
            }
        }
    }
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
