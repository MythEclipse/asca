//! Baseline dense Transformer di crate yang sama, memakai `Mat`, `Param`, `Rng`, dan
//! RMSNorm yang identik dengan ASCA v2.
//!
//! Alasan keberadaannya: klaim efisiensi hanya sah kalau pembanding dijalankan di
//! engine, kernel, dan bahasa yang sama. Di v1 perbandingan ini dilakukan lewat
//! PyTorch sehingga yang terukur adalah framework-nya, bukan arsitekturnya.
//!
//! Blok: `x = x + MHA(RMSNorm(x))`, `x = x + FFN(RMSNorm(x))`, dengan RoPE pada q/k
//! (supaya baseline punya informasi posisi, karena delta rule ASCA memang sekuensial).

use crate::tensor::{Mat, Param, Rng, rmsnorm_rows, rmsnorm_rows_backward};

const EPS: f32 = 1e-6;

pub struct DenseCfg {
    pub vocab: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
}

pub struct DenseLayer {
    pub w_qkv: Param,
    pub b_qkv: Param,
    pub w_o: Param,
    pub b_o: Param,
    pub w1: Param,
    pub b1: Param,
    pub w2: Param,
    pub b2: Param,
}

pub struct DenseModel {
    pub cfg: DenseCfg,
    pub embed: Param,
    pub layers: Vec<DenseLayer>,
    pub head: Param,
}

pub struct AttnCache {
    pub xn: Mat,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    /// keluaran attention mentah sebelum proyeksi `w_o` (dibutuhkan untuk dW_o)
    pub o_raw: Mat,
    pub probs: Vec<f32>, // (n_heads * T * T)
}

pub struct DenseLayerCache {
    pub x_in: Mat,
    pub attn: AttnCache,
    pub x_mid: Mat,
    pub xn2: Mat,
    pub hidden: Mat,
    pub x_out: Mat,
}

pub struct DenseCache {
    pub ids: Vec<u32>,
    pub layers: Vec<DenseLayerCache>,
    pub h_final: Mat,
}

#[inline]
fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

#[inline]
fn dsilu(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

fn rope_apply(x: &mut Mat, n_heads: usize, dh: usize, sign: f32) {
    // sign = +1 forward, -1 inverse (transpose rotasi).
    // PENTING: rotasi harus dilakukan PER HEAD (`off + i`), bukan hanya pada slice
    // head pertama. Versi awal lupa offset head sehingga 3 dari 4 head tidak punya
    // informasi posisi sama sekali — baseline jadi lemah secara tidak adil.
    for t in 0..x.rows {
        let row = x.row_mut(t);
        for hh in 0..n_heads {
            let off = hh * dh;
            let mut i = 0;
            while i + 1 < dh {
                let freq = 1.0 / 10000f32.powf(2.0 * i as f32 / dh as f32);
                let ang = t as f32 * freq * sign;
                let (c, s) = (ang.cos(), ang.sin());
                let a = row[off + i];
                let b = row[off + i + 1];
                row[off + i] = a * c - b * s;
                row[off + i + 1] = a * s + b * c;
                i += 2;
            }
        }
    }
}

impl DenseLayer {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, rng: &mut Rng) -> Self {
        let mut w_qkv = Mat::zeros(d_model, 3 * d_model);
        w_qkv.fill_rand(rng, (1.0 / d_model as f32).sqrt());
        let mut w_o = Mat::zeros(d_model, d_model);
        w_o.fill_rand(rng, (1.0 / d_model as f32).sqrt());
        let mut w1 = Mat::zeros(d_model, d_ff);
        w1.fill_rand(rng, (1.0 / d_model as f32).sqrt());
        let mut w2 = Mat::zeros(d_ff, d_model);
        w2.fill_rand(rng, (1.0 / d_ff as f32).sqrt());
        let _ = n_heads;
        Self {
            w_qkv: Param::new("dense.w_qkv", w_qkv),
            b_qkv: Param::new("dense.b_qkv", Mat::zeros(1, 3 * d_model)),
            w_o: Param::new("dense.w_o", w_o),
            b_o: Param::new("dense.b_o", Mat::zeros(1, d_model)),
            w1: Param::new("dense.w1", w1),
            b1: Param::new("dense.b1", Mat::zeros(1, d_ff)),
            w2: Param::new("dense.w2", w2),
            b2: Param::new("dense.b2", Mat::zeros(1, d_model)),
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![
            &mut self.w_qkv,
            &mut self.b_qkv,
            &mut self.w_o,
            &mut self.b_o,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
        ]
    }

    pub fn params(&self) -> Vec<&Param> {
        vec![
            &self.w_qkv,
            &self.b_qkv,
            &self.w_o,
            &self.b_o,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
        ]
    }
}

impl DenseModel {
    pub fn new(cfg: DenseCfg, rng: &mut Rng) -> Self {
        let mut embed = Mat::zeros(cfg.vocab, cfg.d_model);
        embed.fill_rand(rng, 0.02);
        let mut head = Mat::zeros(cfg.d_model, cfg.vocab);
        head.fill_rand(rng, (1.0 / cfg.d_model as f32).sqrt());
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(DenseLayer::new(cfg.d_model, cfg.n_heads, cfg.d_ff, rng));
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
            v.extend(l.params_mut());
        }
        v
    }

    pub fn params(&self) -> Vec<&Param> {
        let mut v: Vec<&Param> = vec![&self.embed, &self.head];
        for l in self.layers.iter() {
            v.extend(l.params());
        }
        v
    }

    pub fn zero_grads(&mut self) {
        for p in self.params_mut() {
            p.zero_grad();
        }
    }

    pub fn forward(&self, ids: &[u32]) -> (Mat, DenseCache) {
        let t_len = ids.len();
        let d = self.cfg.d_model;
        let h = self.cfg.n_heads;
        let dh = d / h;
        let scale = 1.0 / (dh as f32).sqrt();

        let mut x = Mat::zeros(t_len, d);
        for (t, &id) in ids.iter().enumerate() {
            let row = self.embed.value.row(id as usize);
            for c in 0..d {
                x.set(t, c, row[c]);
            }
        }

        let mut caches = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter() {
            let x_in = x.clone();
            let xn = rmsnorm_rows(&x_in, EPS);
            let mut qkv = xn.matmul(&layer.w_qkv.value);
            qkv.add_row_bias(&layer.b_qkv.value.d);
            let mut q = Mat::zeros(t_len, d);
            let mut k = Mat::zeros(t_len, d);
            let mut v = Mat::zeros(t_len, d);
            for r in 0..t_len {
                let src = qkv.row(r);
                for c in 0..d {
                    q.set(r, c, src[c]);
                    k.set(r, c, src[d + c]);
                    v.set(r, c, src[2 * d + c]);
                }
            }
            rope_apply(&mut q, h, dh, 1.0);
            rope_apply(&mut k, h, dh, 1.0);

            let mut probs = vec![0.0f32; h * t_len * t_len];
            let mut o = Mat::zeros(t_len, d);
            for hh in 0..h {
                let off = hh * dh;
                for t in 0..t_len {
                    // skor kausal
                    let mut row = vec![0.0f32; t + 1];
                    for s in 0..=t {
                        let mut acc = 0.0f32;
                        for c in 0..dh {
                            acc += q.at(t, off + c) * k.at(s, off + c);
                        }
                        row[s] = acc * scale;
                    }
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for z in row.iter_mut() {
                        *z = (*z - max).exp();
                        sum += *z;
                    }
                    for z in row.iter_mut() {
                        *z /= sum;
                    }
                    for s in 0..=t {
                        probs[(hh * t_len + t) * t_len + s] = row[s];
                    }
                    for c in 0..dh {
                        let mut acc = 0.0f32;
                        for s in 0..=t {
                            acc += row[s] * v.at(s, off + c);
                        }
                        o.set(t, off + c, acc);
                    }
                }
            }
            let mut attn_out = o.matmul(&layer.w_o.value);
            attn_out.add_row_bias(&layer.b_o.value.d);
            let mut x_mid = x_in.clone();
            x_mid.axpy(1.0, &attn_out);

            let xn2 = rmsnorm_rows(&x_mid, EPS);
            let mut pre = xn2.matmul(&layer.w1.value);
            pre.add_row_bias(&layer.b1.value.d);
            let mut hidden = Mat::zeros(t_len, self.cfg.d_ff);
            for r in 0..t_len {
                for c in 0..self.cfg.d_ff {
                    hidden.set(r, c, silu(pre.at(r, c)));
                }
            }
            let mut ff = hidden.matmul(&layer.w2.value);
            ff.add_row_bias(&layer.b2.value.d);
            let mut x_out = x_mid.clone();
            x_out.axpy(1.0, &ff);

            x = x_out.clone();
            caches.push(DenseLayerCache {
                x_in,
                attn: AttnCache {
                    xn,
                    q,
                    k,
                    v,
                    o_raw: o,
                    probs,
                },
                x_mid,
                xn2,
                hidden,
                x_out,
            });
        }

        let h_final = rmsnorm_rows(&x, EPS);
        let logits = h_final.matmul(&self.head.value);
        (
            logits,
            DenseCache {
                ids: ids.to_vec(),
                layers: caches,
                h_final,
            },
        )
    }

    pub fn loss(&self, ids: &[u32], targets: &[u32]) -> f32 {
        let (logits, _) = self.forward(ids);
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

    pub fn backward(&mut self, cache: &DenseCache, logits: &Mat, targets: &[u32]) -> f32 {
        let t_len = logits.rows;
        let v = self.cfg.vocab;
        let d = self.cfg.d_model;
        let h = self.cfg.n_heads;
        let dh = d / h;
        let scale = 1.0 / (dh as f32).sqrt();

        let mut dlogits = Mat::zeros(t_len, v);
        let mut loss = 0.0f32;
        for t in 0..t_len {
            let row = logits.row(t);
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for c in 0..v {
                sum += (row[c] - max).exp();
            }
            let lse = max + sum.ln();
            loss += lse - row[targets[t] as usize];
            for c in 0..v {
                dlogits.set(t, c, (row[c] - lse).exp());
            }
            dlogits.set(
                t,
                targets[t] as usize,
                dlogits.at(t, targets[t] as usize) - 1.0,
            );
        }
        loss /= t_len as f32;
        for z in dlogits.d.iter_mut() {
            *z /= t_len as f32;
        }

        let d_head = cache.h_final.transpose().matmul(&dlogits);
        self.head.grad.axpy(1.0, &d_head);
        let mut dx = rmsnorm_rows_backward(
            &cache.layers.last().unwrap().x_out,
            &dlogits.matmul(&self.head.value.transpose()),
            EPS,
        );

        for (li, layer) in self.layers.iter_mut().enumerate().rev() {
            let lc = &cache.layers[li];
            // residual FFN
            let mut d_mid = dx.clone();
            let d_ff_out = dx.clone();
            let d_w2 = lc.hidden.transpose().matmul(&d_ff_out);
            layer.w2.grad.axpy(1.0, &d_w2);
            for r in 0..t_len {
                for c in 0..d {
                    layer.b2.grad.d[c] += d_ff_out.at(r, c);
                }
            }
            let d_hidden = d_ff_out.matmul(&layer.w2.value.transpose());
            let mut d_pre = Mat::zeros(t_len, self.cfg.d_ff);
            for r in 0..t_len {
                for c in 0..self.cfg.d_ff {
                    let pre = {
                        // rekonstruksi pra-aktivasi dari hidden: silu tidak invertibel,
                        // jadi simpan lewat hidden -> gunakan nilai dari cache xn2/w1
                        let mut acc = layer.b1.value.at(0, c);
                        for cc in 0..d {
                            acc += lc.xn2.at(r, cc) * layer.w1.value.at(cc, c);
                        }
                        acc
                    };
                    d_pre.set(r, c, d_hidden.at(r, c) * dsilu(pre));
                }
            }
            let d_w1 = lc.xn2.transpose().matmul(&d_pre);
            layer.w1.grad.axpy(1.0, &d_w1);
            for r in 0..t_len {
                for c in 0..self.cfg.d_ff {
                    layer.b1.grad.d[c] += d_pre.at(r, c);
                }
            }
            d_mid.axpy(
                1.0,
                &rmsnorm_rows_backward(&lc.x_mid, &d_pre.matmul(&layer.w1.value.transpose()), EPS),
            );

            // residual attention
            let mut d_in = d_mid.clone();
            let d_attn_out = d_mid.clone();
            let d_w_o = lc.attn.o_raw.transpose().matmul(&d_attn_out);
            layer.w_o.grad.axpy(1.0, &d_w_o);
            for r in 0..t_len {
                for c in 0..d {
                    layer.b_o.grad.d[c] += d_attn_out.at(r, c);
                }
            }
            let d_o = d_attn_out.matmul(&layer.w_o.value.transpose());

            // attention backward per head
            let mut dq = Mat::zeros(t_len, d);
            let mut dk = Mat::zeros(t_len, d);
            let mut dv = Mat::zeros(t_len, d);
            for hh in 0..h {
                let off = hh * dh;
                for t in 0..t_len {
                    // dp[t,s] = dO_t . v_s
                    let mut dp = vec![0.0f32; t + 1];
                    let mut sum = 0.0f32;
                    for s in 0..=t {
                        let p = lc.attn.probs[(hh * t_len + t) * t_len + s];
                        let mut acc = 0.0f32;
                        for c in 0..dh {
                            acc += d_o.at(t, off + c) * lc.attn.v.at(s, off + c);
                        }
                        dp[s] = acc;
                        sum += p * acc;
                    }
                    let mut dscores = vec![0.0f32; t + 1];
                    for s in 0..=t {
                        let p = lc.attn.probs[(hh * t_len + t) * t_len + s];
                        dscores[s] = p * (dp[s] - sum);
                    }
                    for s in 0..=t {
                        for c in 0..dh {
                            dv.row_mut(s)[off + c] +=
                                lc.attn.probs[(hh * t_len + t) * t_len + s] * d_o.at(t, off + c);
                            dq.row_mut(t)[off + c] += dscores[s] * lc.attn.k.at(s, off + c) * scale;
                            dk.row_mut(s)[off + c] += dscores[s] * lc.attn.q.at(t, off + c) * scale;
                        }
                    }
                }
            }
            // RoPE backward (rotasi balik)
            rope_apply(&mut dq, h, dh, -1.0);
            rope_apply(&mut dk, h, dh, -1.0);

            let mut d_qkv = Mat::zeros(t_len, 3 * d);
            for r in 0..t_len {
                for c in 0..d {
                    d_qkv.set(r, c, dq.at(r, c));
                    d_qkv.set(r, d + c, dk.at(r, c));
                    d_qkv.set(r, 2 * d + c, dv.at(r, c));
                }
            }
            let d_w_qkv = lc.attn.xn.transpose().matmul(&d_qkv);
            layer.w_qkv.grad.axpy(1.0, &d_w_qkv);
            for r in 0..t_len {
                for c in 0..3 * d {
                    layer.b_qkv.grad.d[c] += d_qkv.at(r, c);
                }
            }
            d_in.axpy(
                1.0,
                &rmsnorm_rows_backward(
                    &lc.x_in,
                    &d_qkv.matmul(&layer.w_qkv.value.transpose()),
                    EPS,
                ),
            );
            dx = d_in;
        }

        for (t, &id) in cache.ids.iter().enumerate() {
            for c in 0..d {
                self.embed.grad.set(
                    id as usize,
                    c,
                    self.embed.grad.at(id as usize, c) + dx.at(t, c),
                );
            }
        }
        loss
    }
}

/// Cari `d_ff` dense yang jumlah parameternya paling dekat dengan target.
pub fn dense_matched_to(
    target_params: usize,
    vocab: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    rng_seed: u64,
) -> DenseModel {
    let per_layer_base = 4 * d_model * d_model + 4 * d_model; // qkv + o + bias
    let embed_head = 2 * vocab * d_model;
    let mut best: Option<(usize, usize)> = None; // (jarak, d_ff)
    let mut d_ff = 16;
    while d_ff <= 16 * d_model {
        // per layer: 2 * d_model * d_ff + d_ff + d_model
        let total = embed_head + n_layers * (per_layer_base + 2 * d_model * d_ff + d_ff + d_model);
        let dist = total.abs_diff(target_params);
        if best.map(|(d, _)| dist < d).unwrap_or(true) {
            best = Some((dist, d_ff));
        }
        d_ff += 8;
    }
    let d_ff = best.unwrap().1;
    let mut rng = Rng::new(rng_seed);
    DenseModel::new(
        DenseCfg {
            vocab,
            d_model,
            n_heads,
            n_layers,
            d_ff,
        },
        &mut rng,
    )
}
