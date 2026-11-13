//! Gated delta-rule block: forward + backward tulisan tangan (BPTT).
//!
//! Rekurensi per head (state `S: dh x dh`, `dh = d_model / n_heads`):
//!
//! ```text
//! Sk_t = S_{t-1}^T k_t
//! u_t  = beta_t * (v_t - alpha_t * Sk_t)
//! S_t  = alpha_t * S_{t-1} + k_t u_t^T
//! o_t  = S_t^T q_t
//! ```
//!
//! Backward diturunkan manual dan diverifikasi dengan beda hingga di
//! `tests/gradcheck.rs`. Tidak ada attention, tidak ada softmax, memori state
//! konstan terhadap panjang konteks.

use crate::tensor::{Mat, Param, Rng, sigmoid};

pub struct DeltaCfg {
    pub d_model: usize,
    pub n_heads: usize,
}

pub struct DeltaBlock {
    pub w_qkv: Param,
    pub b_qkv: Param,
    pub w_gate: Param,
    pub b_gate: Param,
    pub w_o: Param,
    pub b_o: Param,
    pub cfg: DeltaCfg,
    pub dh: usize,
}

pub struct DeltaCache {
    pub x: Mat,
    pub qkv: Mat,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub gate: Mat,
    pub alpha: Mat,
    pub beta: Mat,
    pub qn: Mat,
    pub kn: Mat,
    pub o: Mat,
    /// indeks `hh * batch + b`: (seq+1) state per (head, sekuens),
    /// `s_pre[t]` = state sebelum langkah t, `s_pre[seq]` = state akhir
    pub s_pre: Vec<Vec<f32>>,
    pub batch: usize,
    pub seq: usize,
}

impl DeltaBlock {
    pub fn new(cfg: DeltaCfg, rng: &mut Rng) -> Self {
        assert!(
            cfg.d_model % cfg.n_heads == 0,
            "d_model harus habis dibagi n_heads"
        );
        let d = cfg.d_model;
        let h = cfg.n_heads;
        let dh = d / h;
        let mut w_qkv = Mat::zeros(d, 3 * d);
        w_qkv.fill_rand(rng, (1.0 / d as f32).sqrt());
        let mut w_gate = Mat::zeros(d, 2 * h);
        w_gate.fill_rand(rng, (1.0 / d as f32).sqrt());
        let mut w_o = Mat::zeros(d, d);
        w_o.fill_rand(rng, (1.0 / d as f32).sqrt());
        let mut b_gate = Mat::zeros(1, 2 * h);
        // bias gate: alpha logit -2 (mulai lupa pelan), beta logit -1
        for j in 0..h {
            b_gate.set(0, j, -2.0);
            b_gate.set(0, h + j, -1.0);
        }
        Self {
            w_qkv: Param::new("delta.w_qkv", w_qkv),
            b_qkv: Param::new("delta.b_qkv", Mat::zeros(1, 3 * d)),
            w_gate: Param::new("delta.w_gate", w_gate),
            b_gate: Param::new("delta.b_gate", b_gate),
            w_o: Param::new("delta.w_o", w_o),
            b_o: Param::new("delta.b_o", Mat::zeros(1, d)),
            dh,
            cfg,
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![
            &mut self.w_qkv,
            &mut self.b_qkv,
            &mut self.w_gate,
            &mut self.b_gate,
            &mut self.w_o,
            &mut self.b_o,
        ]
    }

    pub fn params(&self) -> Vec<&Param> {
        vec![
            &self.w_qkv,
            &self.b_qkv,
            &self.w_gate,
            &self.b_gate,
            &self.w_o,
            &self.b_o,
        ]
    }

    /// Forward. `x`: (B*T, d_model) berisi `batch` sekuens yang ditumpuk; rekurensi
    /// tetap berjalan per sekuens (state di-reset di batas sekuens), sedangkan semua
    /// perkalian matriks memakai seluruh B*T baris sekaligus.
    pub fn forward(&self, x: &Mat, batch: usize) -> (Mat, DeltaCache) {
        let (t_len, d) = (x.rows, self.cfg.d_model);
        assert!(
            batch >= 1 && t_len % batch == 0,
            "baris harus habis dibagi batch"
        );
        let seq = t_len / batch;
        let h = self.cfg.n_heads;
        let dh = self.dh;

        let mut qkv = crate::prof::time(0, || x.matmul(&self.w_qkv.value));
        qkv.add_row_bias(&self.b_qkv.value.d);

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
        // L2-normalize q dan k per head, sekaligus simpan normanya untuk backward
        let mut qn = Mat::zeros(t_len, h);
        let mut kn = Mat::zeros(t_len, h);
        for r in 0..t_len {
            for hh in 0..h {
                for which in 0..2 {
                    let m = if which == 0 { &mut q } else { &mut k };
                    let row = m.row_mut(r);
                    let seg = &mut row[hh * dh..(hh + 1) * dh];
                    let n = seg.iter().map(|z| z * z).sum::<f32>().sqrt().max(1e-8);
                    for z in seg.iter_mut() {
                        *z /= n;
                    }
                    if which == 0 {
                        qn.set(r, hh, n);
                    } else {
                        kn.set(r, hh, n);
                    }
                }
            }
        }

        let gate = crate::prof::time(0, || {
            let mut g = x.matmul(&self.w_gate.value);
            g.add_row_bias(&self.b_gate.value.d);
            g
        });
        let mut alpha = Mat::zeros(t_len, h);
        let mut beta = Mat::zeros(t_len, h);
        for r in 0..t_len {
            for hh in 0..h {
                alpha.set(r, hh, sigmoid(gate.at(r, hh)));
                beta.set(r, hh, sigmoid(gate.at(r, h + hh)));
            }
        }

        let (o, s_pre) = crate::prof::time(1, || {
            let mut o = Mat::zeros(t_len, d);
            let mut s_pre: Vec<Vec<f32>> = Vec::with_capacity(h * batch);
            for hh in 0..h {
                let off = hh * dh;
                for bi in 0..batch {
                    let mut s = vec![0.0f32; dh * dh];
                    let mut states = Vec::with_capacity((seq + 1) * dh * dh);
                    // `sk` dan `u` dialokasikan sekali per head, bukan per timestep: versi
                    // sebelumnya membuat 2 Vec x T x H per layer-forward.
                    let mut sk = vec![0.0f32; dh];
                    let mut u = vec![0.0f32; dh];
                    for t in 0..seq {
                        let r = bi * seq + t;
                        states.extend_from_slice(&s);
                        let kt = &k.row(r)[off..off + dh];
                        let vt = &v.row(r)[off..off + dh];
                        let a = alpha.at(r, hh);
                        let b = beta.at(r, hh);
                        // Sk = S^T k  (Sk[i] = sum_a S[a][i] k[a])
                        sk.iter_mut().for_each(|z| *z = 0.0);
                        for a_i in 0..dh {
                            let ka = kt[a_i];
                            if ka == 0.0 {
                                continue;
                            }
                            let srow = &s[a_i * dh..(a_i + 1) * dh];
                            for i in 0..dh {
                                sk[i] += srow[i] * ka;
                            }
                        }
                        // u = beta*(v - alpha*Sk); S = alpha*S + k u^T
                        for i in 0..dh {
                            u[i] = b * (vt[i] - a * sk[i]);
                        }
                        for a_i in 0..dh {
                            for i in 0..dh {
                                s[a_i * dh + i] = a * s[a_i * dh + i] + kt[a_i] * u[i];
                            }
                        }
                        // o_t = S^T q
                        let qt = &q.row(r)[off..off + dh];
                        let orow = o.row_mut(r);
                        for i in 0..dh {
                            let mut acc = 0.0;
                            for a_i in 0..dh {
                                acc += s[a_i * dh + i] * qt[a_i];
                            }
                            orow[off + i] = acc;
                        }
                    }
                    states.extend_from_slice(&s);
                    s_pre.push(states);
                }
            }
            (o, s_pre)
        });

        let mut y = crate::prof::time(2, || o.matmul(&self.w_o.value));
        y.add_row_bias(&self.b_o.value.d);

        let cache = DeltaCache {
            x: x.clone(),
            qkv,
            q,
            k,
            v,
            gate,
            alpha,
            beta,
            qn,
            kn,
            o,
            s_pre,
            batch,
            seq,
        };
        (y, cache)
    }

    /// Backward. Mengembalikan `dx` dan mengakumulasi gradien ke parameter.
    /// Satu langkah decode: `x` adalah (1, d_model), `states[hh]` adalah state
    /// `dh x dh` per head yang dibawa antar token. Tidak ada cache yang tumbuh
    /// terhadap panjang konteks — inilah klaim memori ASCA, dan `step` adalah
    /// satu-satunya tempat yang membuktikannya bisa dijalankan.
    pub fn step(&self, x: &Mat, states: &mut [Vec<f32>]) -> Mat {
        let d = self.cfg.d_model;
        let h = self.cfg.n_heads;
        let dh = self.dh;
        assert_eq!(x.rows, 1, "step hanya untuk satu token");
        let mut qkv = x.matmul(&self.w_qkv.value);
        qkv.add_row_bias(&self.b_qkv.value.d);
        let mut gate = x.matmul(&self.w_gate.value);
        gate.add_row_bias(&self.b_gate.value.d);
        let qkv_row = qkv.row(0);
        let gate_row = gate.row(0);
        let mut o = Mat::zeros(1, d);
        let orow = o.row_mut(0);
        let mut sk = vec![0.0f32; dh];
        let mut u = vec![0.0f32; dh];
        for hh in 0..h {
            let off = hh * dh;
            let a = sigmoid(gate_row[hh]);
            let b = sigmoid(gate_row[h + hh]);
            // L2-normalize q dan k per head, sama seperti forward
            let mut qn = [0.0f32; 64];
            let mut kn = [0.0f32; 64];
            let (mut qq, mut kk) = (0.0f32, 0.0f32);
            for i in 0..dh {
                let qv = qkv_row[off + i];
                let kv = qkv_row[d + off + i];
                qq += qv * qv;
                kk += kv * kv;
            }
            // harus identik dengan forward: bagi dengan max(norm, 1e-8)
            let qi = 1.0 / qq.sqrt().max(1e-8);
            let ki = 1.0 / kk.sqrt().max(1e-8);
            for i in 0..dh {
                qn[i] = qkv_row[off + i] * qi;
                kn[i] = qkv_row[d + off + i] * ki;
            }
            let s = &mut states[hh];
            // Sk = S^T k
            sk.iter_mut().for_each(|z| *z = 0.0);
            for a_i in 0..dh {
                let ka = kn[a_i];
                if ka == 0.0 {
                    continue;
                }
                crate::simd::axpy(&mut sk, ka, &s[a_i * dh..(a_i + 1) * dh]);
            }
            // u = beta*(v - alpha*Sk); S = alpha*S + k u^T
            for i in 0..dh {
                u[i] = b * (qkv_row[2 * d + off + i] - a * sk[i]);
            }
            for a_i in 0..dh {
                let ka = kn[a_i];
                if ka == 0.0 {
                    crate::simd::scale(&mut s[a_i * dh..(a_i + 1) * dh], a);
                    continue;
                }
                crate::simd::scale_axpy(&mut s[a_i * dh..(a_i + 1) * dh], a, ka, &u);
            }
            // o = S^T q
            for i in 0..dh {
                let mut acc = 0.0f32;
                for a_i in 0..dh {
                    acc += s[a_i * dh + i] * qn[a_i];
                }
                orow[off + i] = acc;
            }
        }
        let mut y = o.matmul(&self.w_o.value);
        y.add_row_bias(&self.b_o.value.d);
        y
    }

    /// Berapa float state yang dibawa antar token: `n_heads * dh * dh`, konstan.
    pub fn state_floats(&self) -> usize {
        self.cfg.n_heads * self.dh * self.dh
    }

    pub fn backward(&mut self, cache: &DeltaCache, dy: &Mat) -> Mat {
        let (t_len, d) = (cache.x.rows, self.cfg.d_model);
        let h = self.cfg.n_heads;
        let dh = self.dh;

        // y = o @ w_o + b_o
        let d_o = dy.matmul_t(&self.w_o.value);
        // dW_o = o^T @ dy ; db_o = sum(dy)
        let d_w_o = cache.o.matmul_ta(dy);
        self.w_o.grad.axpy(1.0, &d_w_o);
        for r in 0..dy.rows {
            let src = dy.row(r);
            for c in 0..dy.cols {
                self.b_o.grad.d[c] += src[c];
            }
        }

        let mut d_q = Mat::zeros(t_len, d);
        let mut d_k = Mat::zeros(t_len, d);
        let mut d_v = Mat::zeros(t_len, d);
        let mut d_alpha = Mat::zeros(t_len, h);
        let mut d_beta = Mat::zeros(t_len, h);

        for hh in 0..h {
            let off = hh * dh;
            for bi in 0..cache.batch {
                let states = &cache.s_pre[hh * cache.batch + bi];
                let mut ds = vec![0.0f32; dh * dh];
                for t in (0..cache.seq).rev() {
                    let r = bi * cache.seq + t;
                    let s_t = &states[(t + 1) * dh * dh..(t + 2) * dh * dh];
                    let s_prev = &states[t * dh * dh..(t + 1) * dh * dh];
                    let qt = &cache.q.row(r)[off..off + dh];
                    let kt = &cache.k.row(r)[off..off + dh];
                    let vt = &cache.v.row(r)[off..off + dh];
                    let a = cache.alpha.at(r, hh);
                    let b = cache.beta.at(r, hh);
                    let do_t = &d_o.row(r)[off..off + dh];

                    // o_t = S_t^T q_t
                    {
                        let mut dq = vec![0.0f32; dh];
                        for a_i in 0..dh {
                            let mut acc = 0.0;
                            for i in 0..dh {
                                acc += do_t[i] * s_t[a_i * dh + i];
                            }
                            dq[a_i] = acc;
                        }
                        let row = d_q.row_mut(r);
                        for a_i in 0..dh {
                            row[off + a_i] += dq[a_i];
                        }
                        for a_i in 0..dh {
                            for i in 0..dh {
                                ds[a_i * dh + i] += qt[a_i] * do_t[i];
                            }
                        }
                    }

                    // Sk = S_prev^T k ; u = beta*(v - alpha*Sk)
                    // rekonstruksi Sk dan u dari cache
                    let mut sk = vec![0.0f32; dh];
                    for i in 0..dh {
                        let mut acc = 0.0;
                        for a_i in 0..dh {
                            acc += s_prev[a_i * dh + i] * kt[a_i];
                        }
                        sk[i] = acc;
                    }
                    let mut u = vec![0.0f32; dh];
                    for i in 0..dh {
                        u[i] = b * (vt[i] - a * sk[i]);
                    }

                    // S_t = alpha*S_prev + k u^T
                    let mut dalpha = 0.0f32;
                    let mut dbeta = 0.0f32;
                    let mut dk = vec![0.0f32; dh];
                    let mut du = vec![0.0f32; dh];
                    for a_i in 0..dh {
                        for i in 0..dh {
                            dalpha += ds[a_i * dh + i] * s_prev[a_i * dh + i];
                        }
                    }
                    for a_i in 0..dh {
                        let mut acc = 0.0;
                        for i in 0..dh {
                            acc += ds[a_i * dh + i] * u[i];
                        }
                        dk[a_i] += acc;
                    }
                    for i in 0..dh {
                        let mut acc = 0.0;
                        for a_i in 0..dh {
                            acc += ds[a_i * dh + i] * kt[a_i];
                        }
                        du[i] = acc;
                    }
                    let mut ds_prev = vec![0.0f32; dh * dh];
                    for idx in 0..dh * dh {
                        ds_prev[idx] = a * ds[idx];
                    }

                    // u = beta*(v - alpha*Sk)
                    let mut dsk = vec![0.0f32; dh];
                    for i in 0..dh {
                        dbeta += du[i] * (vt[i] - a * sk[i]);
                        d_v.row_mut(r)[off + i] += b * du[i];
                        dsk[i] = -b * a * du[i];
                        dalpha += -b * du[i] * sk[i];
                    }

                    // Sk = S_prev^T k
                    for a_i in 0..dh {
                        let mut acc = 0.0;
                        for i in 0..dh {
                            acc += s_prev[a_i * dh + i] * dsk[i];
                        }
                        dk[a_i] += acc;
                        for i in 0..dh {
                            ds_prev[a_i * dh + i] += kt[a_i] * dsk[i];
                        }
                    }

                    let dk_row = d_k.row_mut(r);
                    for a_i in 0..dh {
                        dk_row[off + a_i] += dk[a_i];
                    }
                    d_alpha.set(r, hh, dalpha);
                    d_beta.set(r, hh, dbeta);
                    ds = ds_prev;
                }
            }
        }

        // Backward L2-normalize q,k per head
        norm_backward(&cache.q, &cache.qn, &mut d_q, dh);
        norm_backward(&cache.k, &cache.kn, &mut d_k, dh);

        // dqkv = concat(dq, dk, dv)
        let mut d_qkv = Mat::zeros(t_len, 3 * d);
        for r in 0..t_len {
            for c in 0..d {
                d_qkv.set(r, c, d_q.at(r, c));
                d_qkv.set(r, d + c, d_k.at(r, c));
                d_qkv.set(r, 2 * d + c, d_v.at(r, c));
            }
        }
        let d_w_qkv = cache.x.transpose().matmul(&d_qkv);
        self.w_qkv.grad.axpy(1.0, &d_w_qkv);
        for r in 0..t_len {
            for c in 0..d_qkv.cols {
                self.b_qkv.grad.d[c] += d_qkv.at(r, c);
            }
        }

        // gate: alpha = sigmoid(g[:, :h]), beta = sigmoid(g[:, h:])
        let mut d_gate = Mat::zeros(t_len, 2 * h);
        for r in 0..t_len {
            for hh in 0..h {
                let al = cache.alpha.at(r, hh);
                let be = cache.beta.at(r, hh);
                d_gate.set(r, hh, d_alpha.at(r, hh) * al * (1.0 - al));
                d_gate.set(r, h + hh, d_beta.at(r, hh) * be * (1.0 - be));
            }
        }
        let d_w_gate = cache.x.transpose().matmul(&d_gate);
        self.w_gate.grad.axpy(1.0, &d_w_gate);
        for r in 0..t_len {
            for c in 0..d_gate.cols {
                self.b_gate.grad.d[c] += d_gate.at(r, c);
            }
        }

        // dx hanya lewat dua jalur: qkv dan gate. `d_o` TIDAK memberi suku langsung
        // ke x — o dihasilkan oleh rekurensi dari q,k,v, dan kontribusinya sudah
        // dialirkan per head di atas.
        let mut dx = d_qkv.matmul(&self.w_qkv.value.transpose());
        dx.axpy(1.0, &d_gate.matmul(&self.w_gate.value.transpose()));
        dx
    }
}

/// Backward L2-normalize per segmen head: `dx = (dy - y (y.dy)) / s`.
fn norm_backward(y: &Mat, norms: &Mat, d: &mut Mat, dh: usize) {
    let (t_len, h) = (y.rows, norms.cols);
    for t in 0..t_len {
        for hh in 0..h {
            let off = hh * dh;
            let s = norms.at(t, hh);
            let row_y = y.row(t);
            let mut dot = 0.0f32;
            {
                let row_d = d.row(t);
                for c in 0..dh {
                    dot += row_y[off + c] * row_d[off + c];
                }
            }
            let row_d = d.row_mut(t);
            for c in 0..dh {
                row_d[off + c] = (row_d[off + c] - row_y[off + c] * dot) / s;
            }
        }
    }
}
