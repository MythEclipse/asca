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

    /// Salinan kolom expert `e` dari w1 (d_model x n_experts*d_ff) -> (d_model x d_ff).
    fn expert_w1(&self, e: usize) -> Mat {
        let d = self.w1.value.rows;
        let d_ff = self.d_ff;
        let mut out = Mat::zeros(d, d_ff);
        for r in 0..d {
            let src = &self.w1.value.row(r)[e * d_ff..(e + 1) * d_ff];
            out.row_mut(r).copy_from_slice(src);
        }
        out
    }

    /// Salinan baris expert `e` dari w2 (n_experts*d_ff x d_model) -> (d_ff x d_model).
    fn expert_w2(&self, e: usize) -> Mat {
        let d = self.w2.value.cols;
        let d_ff = self.d_ff;
        let mut out = Mat::zeros(d_ff, d);
        for r in 0..d_ff {
            let src = self.w2.value.row(e * d_ff + r);
            out.row_mut(r).copy_from_slice(src);
        }
        out
    }

    /// Jalur GPU: tiap expert jadi GEMM biasa setelah barisnya dikumpulkan.
    ///
    /// Versi CPU menghitung `xn @ w1[:, expert]` sebagai loop per kolom dengan `axpy`
    /// per baris token — itu bagian terbesar MoE (15% forward + 23% backward dari total
    /// waktu training) dan tidak pernah menyentuh GPU karena bentuknya bukan GEMM.
    /// Dengan gather -> GEMM -> SiLU -> GEMM -> scatter, seluruh pekerjaan berat menjadi
    /// matmul yang memang sudah punya jalur GPU.
    fn forward_gpu(&self, x: &Mat, _ids: &[u32], buckets: &[Vec<usize>]) -> (Mat, MoeCache) {
        let (t_len, d) = (x.rows, x.cols);
        let d_ff = self.d_ff;
        let xn = crate::prof::time(6, || rmsnorm_rows(x, EPS));
        let mut route = vec![0usize; t_len];
        for (e, rows) in buckets.iter().enumerate() {
            for &t in rows.iter() {
                route[t] = e;
            }
        }
        let mut pre = vec![0.0f32; t_len * d_ff];
        let mut hidden = vec![0.0f32; t_len * d_ff];
        let mut out = Mat::zeros(t_len, d);
        let g = crate::cuda::gpu().unwrap();
        for (e, rows) in buckets.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let xe = g.gather_rows(&xn, rows).expect("gather gagal");
            let w1e = self.expert_w1(e);
            let mut pe = xe.matmul(&w1e);
            let b1 = &self.b1.value.d[e * d_ff..(e + 1) * d_ff];
            for i in 0..pe.rows {
                let row = pe.row_mut(i);
                for j in 0..d_ff {
                    row[j] += b1[j];
                }
            }
            let mut he = pe.clone();
            g.silu_inplace(&mut he).expect("silu gagal");
            let w2e = self.expert_w2(e);
            let mut oe = he.matmul(&w2e);
            let b2 = &self.b2.value.d;
            for i in 0..oe.rows {
                let row = oe.row_mut(i);
                for c in 0..d {
                    row[c] += b2[c];
                }
            }
            g.scatter_rows(&oe, rows, &mut out).expect("scatter gagal");
            for (i, &t) in rows.iter().enumerate() {
                pre[t * d_ff..(t + 1) * d_ff].copy_from_slice(&pe.row(i));
                hidden[t * d_ff..(t + 1) * d_ff].copy_from_slice(&he.row(i));
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
                buckets: buckets.to_vec(),
            },
        )
    }

    pub fn forward(&self, x: &Mat, ids: &[u32]) -> (Mat, MoeCache) {
        let (t_len, d) = (x.rows, x.cols);
        let d_ff = self.d_ff;
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); self.n_experts];
        for t in 0..t_len {
            buckets[hash_route(ids[t], self.n_experts)].push(t);
        }
        // `ASCA_GPU_OFF=moe` memaksa jalur CPU: jalur GPU di bawah memakai gather/scatter
        // per expert, dan itu terbukti bisa lebih lambat daripada CPU kalau biaya
        // salinannya lebih besar dari kerja matmulnya (lihat docs/CUDA.md §3b).
        if crate::cuda::offload("moe") && t_len * d * d_ff >= 1 << 21 {
            return self.forward_gpu(x, ids, &buckets);
        }
        let xn = crate::prof::time(6, || rmsnorm_rows(x, EPS));
        let mut route = vec![0usize; t_len];
        for (e, rows) in buckets.iter().enumerate() {
            for &t in rows.iter() {
                route[t] = e;
            }
        }

        // Tiap expert mendapat buffer kontigu miliknya sendiri. Sebelumnya semua expert
        // menulis ke vec bersama (pre/hidden/out) sehingga loop ini tidak bisa
        // diparalelkan tanpa unsafe; dengan buffer per expert pembagiannya bersih dan
        // hasilnya digabung setelah semua thread selesai. Baris yang ditulis tiap expert
        // tetap sama persis, jadi matematikanya tidak berubah.
        struct Job {
            e: usize,
            rows: Vec<usize>,
            pre: Vec<f32>,
            hidden: Vec<f32>,
            out: Vec<f32>,
        }
        let mut jobs: Vec<Job> = buckets
            .iter()
            .enumerate()
            .map(|(e, rows)| Job {
                e,
                rows: rows.clone(),
                pre: vec![0.0f32; rows.len() * d_ff],
                hidden: vec![0.0f32; rows.len() * d_ff],
                out: vec![0.0f32; rows.len() * d],
            })
            .collect();

        let w1v = &self.w1.value;
        let w2v = &self.w2.value;
        let b1v = &self.b1.value;
        let b2v = &self.b2.value;
        let xnr = &xn;
        std::thread::scope(|s| {
            for job in jobs.iter_mut() {
                s.spawn(move || {
                    let e = job.e;
                    let n = job.rows.len();
                    if n == 0 {
                        return;
                    }
                    let (pre, hidden, out) = (&mut job.pre, &mut job.hidden, &mut job.out);
                    let rows = &job.rows;
                    let b1 = &b1v.d[e * d_ff..(e + 1) * d_ff];
                    for i in 0..n {
                        pre[i * d_ff..(i + 1) * d_ff].copy_from_slice(b1);
                    }
                    // pre = xn @ w1[:, expert]
                    for c in 0..d {
                        let w1row = &w1v.row(c)[e * d_ff..(e + 1) * d_ff];
                        for (i, &t) in rows.iter().enumerate() {
                            let xc = xnr.at(t, c);
                            if xc == 0.0 {
                                continue;
                            }
                            crate::simd::axpy(&mut pre[i * d_ff..(i + 1) * d_ff], xc, w1row);
                        }
                    }
                    for i in 0..n {
                        for j in 0..d_ff {
                            hidden[i * d_ff + j] = silu(pre[i * d_ff + j]);
                        }
                        out[i * d..(i + 1) * d].copy_from_slice(&b2v.d);
                    }
                    // out += hidden @ w2[expert, :]
                    for j in 0..d_ff {
                        let w2row = w2v.row(e * d_ff + j);
                        for i in 0..n {
                            let hj = hidden[i * d_ff + j];
                            if hj == 0.0 {
                                continue;
                            }
                            crate::simd::axpy(&mut out[i * d..(i + 1) * d], hj, w2row);
                        }
                    }
                });
            }
        });

        let mut pre = vec![0.0f32; t_len * d_ff];
        let mut hidden = vec![0.0f32; t_len * d_ff];
        let mut out = Mat::zeros(t_len, d);
        for job in jobs.iter() {
            for (i, &t) in job.rows.iter().enumerate() {
                pre[t * d_ff..(t + 1) * d_ff].copy_from_slice(&job.pre[i * d_ff..(i + 1) * d_ff]);
                hidden[t * d_ff..(t + 1) * d_ff]
                    .copy_from_slice(&job.hidden[i * d_ff..(i + 1) * d_ff]);
                out.row_mut(t).copy_from_slice(&job.out[i * d..(i + 1) * d]);
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
        let mut dxn = Mat::zeros(t_len, d);

        // Sama seperti forward: tiap expert menghitung ke buffer gradiennya sendiri, lalu
        // digabung. Di versi lama semua expert menulis ke w1.grad/w2.grad/b1.grad/b2.grad
        // bersama sehingga loop 32,4% waktu training ini terkunci di satu core.
        struct Job {
            e: usize,
            rows: Vec<usize>,
            dpre: Vec<f32>,
            dw1: Vec<f32>,
            dw2: Vec<f32>,
            db1: Vec<f32>,
            db2: Vec<f32>,
            dxn: Vec<f32>,
        }
        let mut jobs: Vec<Job> = cache
            .buckets
            .iter()
            .enumerate()
            .map(|(e, rows)| Job {
                e,
                rows: rows.clone(),
                dpre: vec![0.0f32; rows.len() * d_ff],
                dw1: vec![0.0f32; d * d_ff],
                dw2: vec![0.0f32; d_ff * d],
                db1: vec![0.0f32; d_ff],
                db2: vec![0.0f32; d],
                dxn: vec![0.0f32; rows.len() * d],
            })
            .collect();

        let w1v = &self.w1.value;
        let w2v = &self.w2.value;
        std::thread::scope(|s| {
            for job in jobs.iter_mut() {
                s.spawn(move || {
                    let e = job.e;
                    let n = job.rows.len();
                    if n == 0 {
                        return;
                    }
                    let rows = &job.rows;
                    let (dpre, dw1, dw2, db1, db2, dxn) = (
                        &mut job.dpre,
                        &mut job.dw1,
                        &mut job.dw2,
                        &mut job.db1,
                        &mut job.db2,
                        &mut job.dxn,
                    );
                    // dW2 += hidden^T dout ; db2 += dout
                    for &t in rows.iter() {
                        let dr = dout.row(t);
                        for j in 0..d_ff {
                            let hj = cache.hidden[t * d_ff + j];
                            if hj == 0.0 {
                                continue;
                            }
                            crate::simd::axpy(&mut dw2[j * d..(j + 1) * d], hj, dr);
                        }
                        for c in 0..d {
                            db2[c] += dr[c];
                        }
                    }
                    // dpre = dout @ w2[expert]^T, lalu dikali turunan SiLU
                    for (i, &t) in rows.iter().enumerate() {
                        let dr = dout.row(t);
                        for j in 0..d_ff {
                            let w2row = w2v.row(e * d_ff + j);
                            dpre[i * d_ff + j] = crate::simd::dot(dr, w2row);
                        }
                    }
                    for (i, &t) in rows.iter().enumerate() {
                        for j in 0..d_ff {
                            let pre = cache.pre[t * d_ff + j];
                            let sg = 1.0 / (1.0 + (-pre).exp());
                            let dh = dpre[i * d_ff + j] * (sg * (1.0 + pre * (1.0 - sg)));
                            db1[j] += dh;
                            dpre[i * d_ff + j] = dh;
                        }
                    }
                    // dW1 += xn^T dpre ; dxn += dpre @ w1[expert]^T
                    for c in 0..d {
                        let w1row = &w1v.row(c)[e * d_ff..(e + 1) * d_ff];
                        let gw = &mut dw1[c * d_ff..(c + 1) * d_ff];
                        for (i, &t) in rows.iter().enumerate() {
                            let xc = cache.xn.at(t, c);
                            if xc == 0.0 {
                                continue;
                            }
                            crate::simd::axpy(gw, xc, &dpre[i * d_ff..(i + 1) * d_ff]);
                        }
                        for i in 0..n {
                            dxn[i * d + c] +=
                                crate::simd::dot(&dpre[i * d_ff..(i + 1) * d_ff], w1row);
                        }
                    }
                });
            }
        });

        // gabungkan hasil per expert
        for job in jobs.iter() {
            let e = job.e;
            for j in 0..d_ff {
                let row = self.w2.grad.row_mut(e * d_ff + j);
                for c in 0..d {
                    row[c] += job.dw2[j * d + c];
                }
            }
            for c in 0..d {
                let row = self.w1.grad.row_mut(c);
                for j in 0..d_ff {
                    row[e * d_ff + j] += job.dw1[c * d_ff + j];
                }
            }
            for j in 0..d_ff {
                self.b1.grad.d[e * d_ff + j] += job.db1[j];
            }
            for c in 0..d {
                self.b2.grad.d[c] += job.db2[c];
            }
            for (i, &t) in job.rows.iter().enumerate() {
                let row = dxn.row_mut(t);
                for c in 0..d {
                    row[c] += job.dxn[i * d + c];
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
    // Catatan: dulu ada field `logits: Mat` di sini yang diisi `Mat::zeros(t_len, vocab)`
    // setiap forward tapi TIDAK PERNAH dibaca siapa pun. Pada batch 8 x seq 256 dengan
    // vocab 12288 itu 100 MB yang dialokasi dan di-nol-kan setiap langkah, tanpa guna.
    // Dihapus.
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
        // Jalur GPU: cross-entropy + softmax backward dihitung dalam satu kernel terpadu
        // per token di GPU. Menghilangkan 25 juta exp/ln di CPU dan transfer 100 MB dlogits.
        let (loss, dlogits) = if crate::cuda::available() && t_len * v >= 65536 {
            if let Some(g) = crate::cuda::gpu() {
                match g.cross_entropy_bwd(logits, targets) {
                    Ok(res) => res,
                    Err(_) => cpu_cross_entropy(logits, targets, v),
                }
            } else {
                cpu_cross_entropy(logits, targets, v)
            }
        } else {
            cpu_cross_entropy(logits, targets, v)
        };

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

fn cpu_cross_entropy(logits: &Mat, targets: &[u32], v: usize) -> (f32, Mat) {
    let t_len = logits.rows;
    let mut dlogits = Mat::zeros(t_len, v);
    let nt = crate::tensor::num_threads();
    let inv = 1.0 / t_len as f32;
    let chunk = t_len.div_ceil(nt.max(1));
    let mut loss_parts = vec![0.0f32; t_len.div_ceil(chunk).max(1)];
    let lp = &mut loss_parts;
    let mut it = lp.iter_mut();
    std::thread::scope(|s| {
        for (ci, dpart) in dlogits.d.chunks_mut(chunk * v).enumerate() {
            let start = ci * chunk;
            let len = dpart.len() / v;
            let lslot = it.next().unwrap();
            let lg = &*logits;
            let tg = targets;
            s.spawn(move || {
                let mut l = 0.0f32;
                for i in 0..len {
                    let t = start + i;
                    let row = &lg.d[t * v..(t + 1) * v];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for c in 0..v {
                        sum += (row[c] - max).exp();
                    }
                    let lse = max + sum.ln();
                    l += lse - row[tg[t] as usize];
                    let drow = &mut dpart[i * v..(i + 1) * v];
                    for c in 0..v {
                        drow[c] = (row[c] - lse).exp() * inv;
                    }
                    drow[tg[t] as usize] -= inv;
                }
                *lslot = l;
            });
        }
    });
    let loss: f32 = loss_parts.iter().sum::<f32>() * inv;
    (loss, dlogits)
}
