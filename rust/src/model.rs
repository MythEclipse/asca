//! Model ASCA v2: byte-level, tanpa attention, tanpa softmax di jalur fitur,
//! hash-routed MoE (tanpa router GEMM, tanpa aux loss, beban seimbang by design).
//!
//! Alur per layer:
//! ```text
//! x = x + DeltaBlock(RMSNorm(x))          // state konstan, O(T)
//! x = x + HashMoE(RMSNorm(x))             // top-1, expert dipilih dari hash(token id)
//! ```
//! Kepala terakhir memproyeksikan ke 256 byte; loss = cross-entropy next-byte.

use crate::delta::{DeltaBlock, DeltaCfg};
use crate::tensor::{AdamW, Mat, Param, Rng, rmsnorm_rows, rmsnorm_rows_backward};

pub const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub n_experts: usize,
    pub d_ff: usize,
}

impl Config {
    pub fn tiny() -> Self {
        Self {
            vocab: 256,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            n_experts: 8,
            d_ff: 128,
        }
    }
}

/// FFN top-1 dengan routing hash pada id token: nol FLOP routing, beban seimbang.
pub struct HashMoe {
    pub w1: Param, // (n_experts * d_ff, d_model) -- dipotong per expert
    pub b1: Param,
    pub w2: Param, // (n_experts * d_ff, d_model)
    pub b2: Param,
    pub n_experts: usize,
    pub d_ff: usize,
}

pub struct MoeCache {
    pub x: Mat,
    pub xn: Mat,
    /// pra-aktivasi SiLU, disimpan supaya backward tidak perlu menghitung ulang
    pub pre: Vec<f32>, // (T * d_ff)
    pub hidden: Vec<f32>, // (T * d_ff) aktivasi setelah silu
    pub route: Vec<usize>,
    /// indeks token per expert; dasar pengelompokan di forward dan backward
    pub buckets: Vec<Vec<usize>>,
}

#[inline]
fn hash_route(id: u32, n_experts: usize) -> usize {
    // campuran 32-bit (murmur-style finalizer), lalu modulo
    let mut h = id.wrapping_mul(0x9E37_79B1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    (h as usize) % n_experts
}

#[inline]
fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

impl HashMoe {
    pub fn new(d_model: usize, d_ff: usize, n_experts: usize, rng: &mut Rng) -> Self {
        let mut w1 = Mat::zeros(d_model, n_experts * d_ff);
        w1.fill_rand(rng, (1.0 / d_model as f32).sqrt());
        let mut w2 = Mat::zeros(n_experts * d_ff, d_model);
        w2.fill_rand(rng, (1.0 / d_ff as f32).sqrt());
        Self {
            w1: Param::new("moe.w1", w1),
            b1: Param::new("moe.b1", Mat::zeros(1, n_experts * d_ff)),
            w2: Param::new("moe.w2", w2),
            b2: Param::new("moe.b2", Mat::zeros(1, d_model)),
            n_experts,
            d_ff,
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2]
    }

    pub fn params(&self) -> Vec<&Param> {
        vec![&self.w1, &self.b1, &self.w2, &self.b2]
    }

    /// Forward dengan pengelompokan per expert.
    ///
    /// Versi per-token membaca blok bobot satu expert berulang kali secara acak
    /// (token berganti expert di setiap baris). Dengan mengelompokkan token per
    /// expert dan menaruh loop bobot di luar, satu baris `w1`/`w2` dibaca sekali
    /// lalu dipakai untuk semua token expert itu — pola GEMM yang sebenarnya.
    pub fn forward(&self, x: &Mat, ids: &[u32]) -> (Mat, MoeCache) {
        let (t_len, d) = (x.rows, x.cols);
        let d_ff = self.d_ff;
        let xn = crate::prof::time(6, || rmsnorm_rows(x, EPS));
        let mut route = vec![0usize; t_len];
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); self.n_experts];
        for t in 0..t_len {
            let e = hash_route(ids[t], self.n_experts);
            route[t] = e;
            buckets[e].push(t);
        }
        let mut pre = vec![0.0f32; t_len * d_ff];
        let mut hidden = vec![0.0f32; t_len * d_ff];
        let mut out = Mat::zeros(t_len, d);

        for e in 0..self.n_experts {
            let rows = &buckets[e];
            if rows.is_empty() {
                continue;
            }
            let b1 = &self.b1.value.d[e * d_ff..(e + 1) * d_ff];
            for &t in rows.iter() {
                pre[t * d_ff..(t + 1) * d_ff].copy_from_slice(b1);
            }
            // pre = xn @ w1[:, expert]
            for c in 0..d {
                let w1row = &self.w1.value.row(c)[e * d_ff..(e + 1) * d_ff];
                for &t in rows.iter() {
                    let xc = xn.at(t, c);
                    if xc == 0.0 {
                        continue;
                    }
                    let prow = &mut pre[t * d_ff..(t + 1) * d_ff];
                    crate::simd::axpy(prow, xc, w1row);
                }
            }
            for &t in rows.iter() {
                let prow = &pre[t * d_ff..(t + 1) * d_ff];
                let hrow = &mut hidden[t * d_ff..(t + 1) * d_ff];
                for j in 0..d_ff {
                    hrow[j] = silu(prow[j]);
                }
                out.row_mut(t).copy_from_slice(&self.b2.value.d);
            }
            // out += hidden @ w2[expert, :]
            for j in 0..d_ff {
                let w2row = self.w2.value.row(e * d_ff + j);
                for &t in rows.iter() {
                    let hj = hidden[t * d_ff + j];
                    if hj == 0.0 {
                        continue;
                    }
                    let orow = out.row_mut(t);
                    crate::simd::axpy(orow, hj, w2row);
                }
            }
        }
        (
            out,
            MoeCache {
                x: x.clone(),
                xn,
                pre,
                hidden,
                route,
                buckets,
            },
        )
    }

    /// Satu langkah decode untuk satu token.
    pub fn step(&self, x: &Mat, id: u32) -> Mat {
        let d = x.cols;
        let d_ff = self.d_ff;
        let xn = rmsnorm_rows(x, EPS);
        let e = hash_route(id, self.n_experts);
        let xr = xn.row(0);
        let mut pre = self.b1.value.d[e * d_ff..(e + 1) * d_ff].to_vec();
        for c in 0..d {
            let xc = xr[c];
            if xc == 0.0 {
                continue;
            }
            crate::simd::axpy(
                &mut pre,
                xc,
                &self.w1.value.row(c)[e * d_ff..(e + 1) * d_ff],
            );
        }
        for p in pre.iter_mut() {
            *p = silu(*p);
        }
        let mut out = Mat::zeros(1, d);
        let orow = out.row_mut(0);
        orow.copy_from_slice(&self.b2.value.d);
        for j in 0..d_ff {
            crate::simd::axpy(orow, pre[j], self.w2.value.row(e * d_ff + j));
        }
        out
    }

    pub fn backward(&mut self, cache: &MoeCache, dout: &Mat) -> Mat {
        let (t_len, d) = (cache.x.rows, cache.x.cols);
        let d_ff = self.d_ff;
        let mut dpre = vec![0.0f32; t_len * d_ff];
        let mut dxn = Mat::zeros(t_len, d);

        for e in 0..self.n_experts {
            let rows = &cache.buckets[e];
            if rows.is_empty() {
                continue;
            }
            // dW2 += hidden^T dout ; db2 += dout
            for j in 0..d_ff {
                let g2row = &mut self.w2.grad.row_mut(e * d_ff + j);
                for &t in rows.iter() {
                    let hj = cache.hidden[t * d_ff + j];
                    if hj == 0.0 {
                        continue;
                    }
                    crate::simd::axpy(g2row, hj, dout.row(t));
                }
            }
            for &t in rows.iter() {
                let dr = dout.row(t);
                for c in 0..d {
                    self.b2.grad.d[c] += dr[c];
                }
            }
            // dpre = dout @ w2[expert]^T, lalu dikali turunan SiLU
            for j in 0..d_ff {
                let w2row = self.w2.value.row(e * d_ff + j);
                for &t in rows.iter() {
                    dpre[t * d_ff + j] = crate::simd::dot(dout.row(t), w2row);
                }
            }
            for &t in rows.iter() {
                for j in 0..d_ff {
                    let pre = cache.pre[t * d_ff + j];
                    let s = 1.0 / (1.0 + (-pre).exp());
                    let dh = dpre[t * d_ff + j] * (s * (1.0 + pre * (1.0 - s)));
                    self.b1.grad.d[e * d_ff + j] += dh;
                    dpre[t * d_ff + j] = dh;
                }
            }
            // dW1 += xn^T dpre ; dxn += dpre @ w1[expert]^T
            for c in 0..d {
                let g1row = &mut self.w1.grad.row_mut(c)[e * d_ff..(e + 1) * d_ff];
                for &t in rows.iter() {
                    let xc = cache.xn.at(t, c);
                    if xc == 0.0 {
                        continue;
                    }
                    let drow = &dpre[t * d_ff..(t + 1) * d_ff];
                    crate::simd::axpy(g1row, xc, drow);
                }
                let w1row = &self.w1.value.row(c)[e * d_ff..(e + 1) * d_ff];
                for &t in rows.iter() {
                    let drow = &dpre[t * d_ff..(t + 1) * d_ff];
                    dxn.row_mut(t)[c] += crate::simd::dot(drow, w1row);
                }
            }
        }
        crate::prof::time(6, || rmsnorm_rows_backward(&cache.x, &dxn, EPS))
    }
}

pub struct Layer {
    pub delta: DeltaBlock,
    pub moe: HashMoe,
}

pub struct Model {
    pub cfg: Config,
    pub embed: Param,
    pub layers: Vec<Layer>,
    pub head: Param,
}

pub struct LayerCache {
    pub x_in: Mat,
    pub delta_cache: crate::delta::DeltaCache,
    pub x_mid: Mat,
    pub moe_cache: MoeCache,
    pub x_out: Mat,
}

pub struct DecodeState {
    /// per layer, per head: state `dh x dh`
    pub delta: Vec<Vec<Vec<f32>>>,
}

pub struct ModelCache {
    pub ids: Vec<u32>,
    pub x0: Mat,
    pub layers: Vec<LayerCache>,
    pub h_final: Mat,
    pub logits: Mat,
}

impl Model {
    pub fn new(cfg: Config, rng: &mut Rng) -> Self {
        let mut embed = Mat::zeros(cfg.vocab, cfg.d_model);
        embed.fill_rand(rng, 0.02);
        let mut head = Mat::zeros(cfg.d_model, cfg.vocab);
        head.fill_rand(rng, (1.0 / cfg.d_model as f32).sqrt());
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
                delta: DeltaBlock::new(
                    DeltaCfg {
                        d_model: cfg.d_model,
                        n_heads: cfg.n_heads,
                    },
                    rng,
                ),
                moe: HashMoe::new(cfg.d_model, cfg.d_ff, cfg.n_experts, rng),
            });
        }
        Self {
            cfg,
            embed: Param::new("embed", embed),
            layers,
            head: Param::new("head", head),
        }
    }

    pub fn num_params(&self) -> usize {
        self.params().iter().map(|p| p.numel()).sum()
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut v: Vec<&mut Param> = vec![&mut self.embed, &mut self.head];
        for l in self.layers.iter_mut() {
            v.extend(l.delta.params_mut());
            v.extend(l.moe.params_mut());
        }
        v
    }

    pub fn params(&self) -> Vec<&Param> {
        let mut v: Vec<&Param> = vec![&self.embed, &self.head];
        for l in self.layers.iter() {
            v.extend(l.delta.params());
            v.extend(l.moe.params());
        }
        v
    }

    pub fn zero_grads(&mut self) {
        for p in self.params_mut() {
            p.zero_grad();
        }
    }

    /// State decode ASCA: untuk setiap layer, satu matriks `dh x dh` per head.
    /// Ukurannya tetap, berapa pun panjang konteksnya.
    pub fn new_decode_state(&self) -> DecodeState {
        DecodeState {
            delta: self
                .layers
                .iter()
                .map(|l| vec![vec![0.0f32; l.delta.dh * l.delta.dh]; self.cfg.n_heads])
                .collect(),
        }
    }

    /// Decode satu token: logits `(1, vocab)`, state diperbarui di tempat.
    /// Paritasnya terhadap `forward` diuji di `tests/decode_parity.rs`.
    pub fn step(&self, id: u32, st: &mut DecodeState) -> Mat {
        let d = self.cfg.d_model;
        let mut x = Mat::zeros(1, d);
        x.row_mut(0)
            .copy_from_slice(self.embed.value.row(id as usize));
        for (li, layer) in self.layers.iter().enumerate() {
            // Sama seperti forward: delta menerima input mentah (tidak menormalkan),
            // MoE menormalkan sendiri di dalam.
            let dy = layer.delta.step(&x, &mut st.delta[li]);
            let mut x_mid = x.clone();
            x_mid.axpy(1.0, &dy);
            let dm = layer.moe.step(&x_mid, id);
            let mut x_out = x_mid.clone();
            x_out.axpy(1.0, &dm);
            x = x_out;
        }
        let h_final = rmsnorm_rows(&x, EPS);
        h_final.matmul(&self.head.value)
    }

    pub fn forward(&self, ids: &[u32], batch: usize) -> (Mat, ModelCache) {
        let t_len = ids.len();
        let d = self.cfg.d_model;
        let mut x0 = Mat::zeros(t_len, d);
        for (t, &id) in ids.iter().enumerate() {
            let row = self.embed.value.row(id as usize);
            for c in 0..d {
                x0.set(t, c, row[c]);
            }
        }
        let mut x = x0.clone();
        let mut layer_caches = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter() {
            let x_in = x.clone();
            let (dy, delta_cache) = layer.delta.forward(&x_in, batch);
            let mut x_mid = x_in.clone();
            x_mid.axpy(1.0, &dy);
            let (dm, moe_cache) = crate::prof::time(4, || layer.moe.forward(&x_mid, ids));
            let mut x_out = x_mid.clone();
            x_out.axpy(1.0, &dm);
            x = x_out.clone();
            layer_caches.push(LayerCache {
                x_in,
                delta_cache,
                x_mid,
                moe_cache,
                x_out,
            });
        }
        let h_final = crate::prof::time(6, || rmsnorm_rows(&x, EPS));
        let logits = crate::prof::time(7, || h_final.matmul(&self.head.value));
        (
            logits,
            ModelCache {
                ids: ids.to_vec(),
                x0,
                layers: layer_caches,
                h_final,
                logits: Mat::zeros(t_len, self.cfg.vocab),
            },
        )
    }

    /// Loss saja (untuk gradcheck / evaluasi), tanpa menyentuh gradien.
    pub fn loss(&self, ids: &[u32], targets: &[u32]) -> f32 {
        let (logits, _) = self.forward(ids, 1);
        let v = self.cfg.vocab;
        let mut loss = 0.0f32;
        for t in 0..logits.rows {
            let row = logits.row(t);
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for c in 0..v {
                sum += (row[c] - max).exp();
            }
            loss += max + sum.ln() - row[targets[t] as usize];
        }
        loss / logits.rows as f32
    }

    /// Backward + cross-entropy. `targets[t]` = byte berikutnya.
    /// Mengembalikan loss rata-rata per token dan mengakumulasi gradien.
    pub fn backward(&mut self, cache: &ModelCache, logits: &Mat, targets: &[u32]) -> f32 {
        let t_len = logits.rows;
        let v = self.cfg.vocab;
        let mut dlogits = Mat::zeros(t_len, v);
        let mut loss = 0.0f32;
        // Softmax atas vocab (bisa 8192) lewat slice baris, bukan `set()`/`at()` per
        // elemen: versi lama melakukan 4,2 juta akses ber-bounds-check per langkah.
        for t in 0..t_len {
            let row = logits.row(t);
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for c in 0..v {
                sum += (row[c] - max).exp();
            }
            let lse = max + sum.ln();
            loss += lse - row[targets[t] as usize];
            let drow = dlogits.row_mut(t);
            for c in 0..v {
                drow[c] = (row[c] - lse).exp();
            }
            drow[targets[t] as usize] -= 1.0;
        }
        loss /= t_len as f32;
        for x in dlogits.d.iter_mut() {
            *x /= t_len as f32;
        }

        // head
        let d_head = crate::prof::time(9, || cache.h_final.matmul_ta(&dlogits));
        self.head.grad.axpy(1.0, &d_head);
        let dh = crate::prof::time(9, || dlogits.matmul_t(&self.head.value));

        let mut dx = rmsnorm_rows_backward(&cache.layers.last().unwrap().x_out, &dh, EPS);
        for (li, layer) in self.layers.iter_mut().enumerate().rev() {
            let lc = &cache.layers[li];
            // residual kedua: x_out = x_mid + moe(...)
            let mut d_mid = dx.clone();
            let d_moe = crate::prof::time(5, || layer.moe.backward(&lc.moe_cache, &dx));
            d_mid.axpy(1.0, &d_moe);
            // residual pertama: x_mid = x_in + delta(...)
            let mut d_in = d_mid.clone();
            let d_delta = crate::prof::time(3, || layer.delta.backward(&lc.delta_cache, &d_mid));
            d_in.axpy(1.0, &d_delta);
            dx = d_in;
        }
        // embedding
        for (t, &id) in cache.ids.iter().enumerate() {
            let src = dx.row(t);
            let dst = self.embed.grad.row_mut(id as usize);
            for c in 0..self.cfg.d_model {
                dst[c] += src[c];
            }
        }
        let _ = &cache.x0;
        loss
    }
}

/// Langkah optimasi untuk semua parameter.
pub fn step(model: &mut Model, opt: &mut AdamW) {
    for p in model.params_mut() {
        opt.update(p);
    }
}
