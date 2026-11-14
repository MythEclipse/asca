//! Tensor minimal + RNG + AdamW. Tanpa dependensi eksternal.
//!
//! Semua matriks row-major `rows x cols`. Bobot disimpan dengan orientasi
//! `(in, out)` sehingga forward adalah `x.matmul(w)` tanpa transpose.

#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub d: Vec<f32>,
}

// ------------------------------------------------------------------ paralelisme

/// Jumlah thread kerja. Bisa dikurangi lewat `ASCA_THREADS` untuk A/B.
pub fn num_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        if let Ok(v) = std::env::var("ASCA_THREADS") {
            if let Ok(n) = v.parse::<usize>() {
                return n.max(1);
            }
        }
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

/// Ambang pekerjaan (MAC) sebelum baris dipecah ke beberapa thread. Di bawah ambang ini
/// biaya membuat thread lebih besar daripada matmul-nya sendiri - dan langkah training
/// punya banyak matmul kecil (per head, per expert) yang justru akan melambat.
const PAR_MIN_MAC: usize = 1 << 18;
/// Ambang jumlah elemen sebelum update AdamW dipecah ke beberapa thread.
const PAR_MIN_ADAM: usize = 1 << 14;

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            d: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, d: Vec<f32>) -> Self {
        assert_eq!(d.len(), rows * cols, "ukuran data tidak cocok dengan shape");
        Self { rows, cols, d }
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.d[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.d[r * self.cols + c] = v;
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.d[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.d[r * self.cols..(r + 1) * self.cols]
    }

    pub fn fill_rand(&mut self, rng: &mut Rng, scale: f32) {
        for v in self.d.iter_mut() {
            *v = rng.normal() * scale;
        }
    }

    pub fn fill_const(&mut self, v: f32) {
        self.d.fill(v);
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.d.len()
    }

    pub fn zero_grad_like(&self) -> Self {
        Self::zeros(self.rows, self.cols)
    }

    /// `self (rows x K) @ other (K x cols) -> (rows x cols)`.
    pub fn matmul(&self, other: &Mat) -> Mat {
        // Jalur GPU hanya dipakai kalau ukurannya cukup besar; di bawah ambang itu
        // biaya salin H2D/D2H melebihi matmul CPU-nya sendiri.
        if crate::cuda::available() && crate::cuda::worth_offload(self.rows, self.cols, other.cols)
        {
            if let Some(g) = crate::cuda::gpu() {
                if let Ok(out) = g.matmul(self, other) {
                    return out;
                }
            }
        }
        assert_eq!(self.cols, other.rows, "matmul: inner dim tidak cocok");
        let (n, k, m) = (self.rows, self.cols, other.cols);
        let mut out = Mat::zeros(n, m);
        let (sd, od) = (&self.d, &other.d);
        let nt = num_threads();
        let mac = n.saturating_mul(m).saturating_mul(k);
        if nt <= 1 || mac < PAR_MIN_MAC || n < nt * 2 {
            for i in 0..n {
                let a = &sd[i * k..(i + 1) * k];
                let o = &mut out.d[i * m..(i + 1) * m];
                for p in 0..k {
                    let ap = a[p];
                    if ap == 0.0 {
                        continue;
                    }
                    crate::simd::axpy(o, ap, &od[p * m..(p + 1) * m]);
                }
            }
            return out;
        }
        // `chunks_mut` membagi baris tanpa tumpang tindih, jadi setiap thread punya
        // potongan keluaran sendiri - tidak perlu unsafe dan tidak ada aliasing.
        let chunk = n.div_ceil(nt);
        std::thread::scope(|s| {
            for (ci, part) in out.d.chunks_mut(chunk * m).enumerate() {
                let start = ci * chunk;
                let len = part.len() / m;
                s.spawn(move || {
                    for i in 0..len {
                        let gi = start + i;
                        let a = &sd[gi * k..(gi + 1) * k];
                        let o = &mut part[i * m..(i + 1) * m];
                        for p in 0..k {
                            let ap = a[p];
                            if ap == 0.0 {
                                continue;
                            }
                            crate::simd::axpy(o, ap, &od[p * m..(p + 1) * m]);
                        }
                    }
                });
            }
        });
        out
    }

    /// `self^T @ other` dengan self (K x M) dan other (K x N) -> (M x N).
    /// Sama hasilnya dengan `self.transpose().matmul(other)`, tetapi tanpa
    /// mengalokasi dan menyalin matriks transpose lebih dulu.
    pub fn matmul_ta(&self, other: &Mat) -> Mat {
        // Jalur CPU di bawah tidak punya versi GPU. Untuk bentuk besar, transpose lalu
        // serahkan ke `matmul` supaya offload GPU tetap terpakai — tanpa ini, mengganti
        // `a.transpose().matmul(b)` dengan `a.matmul_ta(b)` justru memperlambat matmul
        // besar (terukur: dense 12.313 -> 9.430 tok/s).
        if crate::cuda::available()
            && crate::cuda::worth_offload(self.rows, self.cols, other.cols)
            && self.rows * self.cols >= 1024
        {
            return self.transpose().matmul(other);
        }
        assert_eq!(self.rows, other.rows, "matmul_ta: dim tidak cocok");
        let (k, m, n) = (self.rows, self.cols, other.cols);
        let mut out = Mat::zeros(m, n);
        for p in 0..k {
            let a = self.row(p);
            let b = other.row(p);
            for i in 0..m {
                let ai = a[i];
                if ai == 0.0 {
                    continue;
                }
                crate::simd::axpy(out.row_mut(i), ai, b);
            }
        }
        out
    }

    /// `self (rows x K) @ other^T` dengan other (cols x K) -> (rows x cols).
    pub fn matmul_t(&self, other: &Mat) -> Mat {
        // Sama seperti matmul_ta: jalur CPU di bawah tidak punya versi GPU. Untuk bentuk
        // besar, transpose lalu serahkan ke `matmul`. Tanpa ini, `dlogits.matmul_t(head)`
        // (512 x 8192 x 128 = 537 juta MAC) jalan di CPU padahal GPU tersedia.
        if crate::cuda::available()
            && crate::cuda::worth_offload(self.rows, self.cols, other.rows)
            && self.rows * self.cols >= 1024
        {
            return self.matmul(&other.transpose());
        }
        assert_eq!(self.cols, other.cols, "matmul_t: dim tidak cocok");
        let (n, m) = (self.rows, other.rows);
        let mut out = Mat::zeros(n, m);
        for i in 0..n {
            let a = self.row(i);
            let o = out.row_mut(i);
            for j in 0..m {
                o[j] = crate::simd::dot(a, other.row(j));
            }
        }
        out
    }

    /// Akumulasi `self += alpha * other` (bentuk harus sama).
    pub fn axpy(&mut self, alpha: f32, other: &Mat) {
        assert_eq!(self.d.len(), other.d.len(), "axpy: bentuk berbeda");
        for (a, b) in self.d.iter_mut().zip(other.d.iter()) {
            *a += alpha * b;
        }
    }

    /// `out = self @ other^T` untuk gradien: dW = x^T @ dy  =>  pakai `x.matmul_t(dy)`
    /// dengan hasil (x.cols x dy.cols). Di sini kita butuh x^T @ dy, jadi transpose
    /// dilakukan eksplisit lewat `transpose()`.
    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.at(i, j));
            }
        }
        out
    }

    /// Tambah bias per kolom: `self[r][c] += b[c]`.
    pub fn add_row_bias(&mut self, b: &[f32]) {
        assert_eq!(b.len(), self.cols);
        let cols = self.cols;
        for r in 0..self.rows {
            let row = self.row_mut(r);
            for c in 0..cols {
                row[c] += b[c];
            }
        }
    }

    pub fn sum(&self) -> f32 {
        self.d.iter().sum()
    }

    pub fn max_abs(&self) -> f32 {
        self.d.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// RMSNorm per baris (tanpa bobot, untuk kesederhanaan verifikasi gradien).
pub fn rmsnorm_rows(x: &Mat, eps: f32) -> Mat {
    if crate::cuda::available() && x.d.len() >= 65536 {
        if let Some(g) = crate::cuda::gpu() {
            if let Ok((y, _)) = g.rmsnorm(x, eps) {
                return y;
            }
        }
    }
    let mut out = Mat::zeros(x.rows, x.cols);
    let cols = x.cols;
    for r in 0..x.rows {
        let row = x.row(r);
        let ms = crate::simd::dot(row, row) / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let o = out.row_mut(r);
        for c in 0..cols {
            o[c] = row[c] * inv;
        }
    }
    out
}

/// Backward RMSNorm: `dx = inv * (dy - x_hat * mean(dy * x_hat))`.
pub fn rmsnorm_rows_backward(x: &Mat, dy: &Mat, eps: f32) -> Mat {
    if crate::cuda::available() && x.d.len() >= 65536 {
        if let Some(g) = crate::cuda::gpu() {
            if let Ok(dx) = g.rmsnorm_bwd(x, dy, eps) {
                return dx;
            }
        }
    }
    let mut dx = Mat::zeros(x.rows, x.cols);
    let cols = x.cols;
    for r in 0..x.rows {
        let row = x.row(r);
        let ms = crate::simd::dot(row, row) / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let dyr = dy.row(r);
        let mut mean = crate::simd::dot(dyr, row) * inv;
        mean /= cols as f32;
        let dxr = dx.row_mut(r);
        for c in 0..cols {
            dxr[c] = inv * (dyr[c] - row[c] * inv * mean);
        }
    }
    dx
}

/// RNG deterministik (xorshift128+), tanpa dependensi.
#[derive(Clone, Debug)]
pub struct Rng {
    s0: u64,
    s1: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let s0 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let s1 = (seed ^ 0xDEAD_BEEF_CAFE_F00D).wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1;
        Self { s0, s1 }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s0;
        let y = self.s1;
        self.s0 = y;
        x ^= x << 23;
        self.s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        self.s1.wrapping_add(y)
    }

    /// Uniform [0,1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Normal standar (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Satu parameter beserta gradien dan momen Adam.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub value: Mat,
    pub grad: Mat,
    pub m: Mat,
    pub v: Mat,
}

impl Param {
    pub fn new(name: impl Into<String>, value: Mat) -> Self {
        let grad = Mat::zeros(value.rows, value.cols);
        Self {
            name: name.into(),
            m: grad.clone(),
            v: grad.clone(),
            value,
            grad,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad.d.fill(0.0);
    }

    pub fn numel(&self) -> usize {
        self.value.d.len()
    }
}

/// AdamW dengan weight decay terpisah (decoupled), seperti di PyTorch.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    pub step: u32,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            wd: 0.01,
            step: 0,
        }
    }

    pub fn update(&mut self, p: &mut Param) {
        self.step += 1;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);
        let lr = self.lr * (bc1 / bc2).sqrt();
        // Loop-invariant: versi lama memanggil bc2.sqrt() untuk SETIAP elemen, yaitu
        // 12,7 juta akar kuadrat per langkah pada model 12,7 juta parameter.
        let eps_t = self.eps * bc2.sqrt();
        let wd = self.lr * self.wd;
        let n = p.value.d.len();
        // Jalur utama: momen tinggal di VRAM, jadi hanya nilai + gradien yang naik dan
        // nilai yang turun (3 salinan, bukan 7).
        if crate::cuda::offload_opt_in("adamw") && n >= 32768 {
            if let Some(g) = crate::cuda::gpu() {
                let key = crate::cuda::param_key(&p.m.d);
                if g.adamw_resident(
                    key, &mut p.value.d, &p.grad.d, &p.m.d, &p.v.d, lr, b1, b2, eps_t, wd,
                )
                .is_ok()
                {
                    return;
                }
            }
        }
        // Jalur cadangan: momen di host mungkin basi kalau jalur resident sempat dipakai.
        if let Some((mm, vv)) = crate::cuda::param_moments(crate::cuda::param_key(&p.m.d)) {
            p.m.d.copy_from_slice(&mm);
            p.v.d.copy_from_slice(&vv);
        }
        if crate::cuda::offload_opt_in("adamw") && n >= 32768 {
            if let Some(g) = crate::cuda::gpu() {
                if g.adamw_update(
                    &mut p.value.d,
                    &p.grad.d,
                    &mut p.m.d,
                    &mut p.v.d,
                    lr,
                    b1,
                    b2,
                    eps_t,
                    wd,
                )
                .is_ok()
                {
                    return;
                }
            }
        }
        let nt = num_threads();
        let chunk = n.div_ceil(nt.max(1));
        let body = |vpart: &mut [f32], mpart: &mut [f32], vvpart: &mut [f32], gpart: &[f32]| {
            for i in 0..vpart.len() {
                let g = gpart[i];
                mpart[i] = b1 * mpart[i] + (1.0 - b1) * g;
                vvpart[i] = b2 * vvpart[i] + (1.0 - b2) * g * g;
                let upd = mpart[i] / (vvpart[i].sqrt() + eps_t);
                vpart[i] -= wd * vpart[i];
                vpart[i] -= lr * upd;
            }
        };
        // Update AdamW sepenuhnya independen per elemen, jadi dipecah ke beberapa thread.
        // Ambang menjaga parameter kecil (bias) tidak membayar biaya pembuatan thread.
        if nt <= 1 || n < PAR_MIN_ADAM {
            body(&mut p.value.d, &mut p.m.d, &mut p.v.d, &p.grad.d);
            return;
        }
        let mut vi = p.value.d.chunks_mut(chunk);
        let mut mi = p.m.d.chunks_mut(chunk);
        let mut vv = p.v.d.chunks_mut(chunk);
        let gi = p.grad.d.chunks(chunk);
        std::thread::scope(|s| {
            for (((vpart, mpart), vvpart), gpart) in
                vi.by_ref().zip(mi.by_ref()).zip(vv.by_ref()).zip(gi)
            {
                let body = &body;
                s.spawn(move || body(vpart, mpart, vvpart, gpart));
            }
        });
    }
}
