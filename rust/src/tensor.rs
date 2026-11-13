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
        for i in 0..n {
            let a = self.row(i);
            let o = out.row_mut(i);
            for p in 0..k {
                let ap = a[p];
                if ap == 0.0 {
                    continue;
                }
                let b = other.row(p);
                for j in 0..m {
                    o[j] += ap * b[j];
                }
            }
        }
        out
    }

    /// `self^T @ other` dengan self (K x M) dan other (K x N) -> (M x N).
    /// Sama hasilnya dengan `self.transpose().matmul(other)`, tetapi tanpa
    /// mengalokasi dan menyalin matriks transpose lebih dulu.
    pub fn matmul_ta(&self, other: &Mat) -> Mat {
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
                let o = out.row_mut(i);
                for j in 0..n {
                    o[j] += ai * b[j];
                }
            }
        }
        out
    }

    /// `self (rows x K) @ other^T` dengan other (cols x K) -> (rows x cols).
    pub fn matmul_t(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.cols, "matmul_t: dim tidak cocok");
        let (n, k, m) = (self.rows, self.cols, other.rows);
        let mut out = Mat::zeros(n, m);
        for i in 0..n {
            let a = self.row(i);
            let o = out.row_mut(i);
            for j in 0..m {
                let b = other.row(j);
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[p] * b[p];
                }
                o[j] = s;
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
    let mut out = Mat::zeros(x.rows, x.cols);
    let cols = x.cols;
    for r in 0..x.rows {
        let row = x.row(r);
        let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
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
    let mut dx = Mat::zeros(x.rows, x.cols);
    let cols = x.cols;
    for r in 0..x.rows {
        let row = x.row(r);
        let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let dyr = dy.row(r);
        let mut mean = 0.0f32;
        for c in 0..cols {
            mean += dyr[c] * row[c] * inv;
        }
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
        for i in 0..p.value.d.len() {
            let g = p.grad.d[i];
            p.m.d[i] = b1 * p.m.d[i] + (1.0 - b1) * g;
            p.v.d[i] = b2 * p.v.d[i] + (1.0 - b2) * g * g;
            let update = p.m.d[i] / (p.v.d[i].sqrt() + self.eps * bc2.sqrt());
            // weight decay decoupled hanya untuk bobot, bukan bias (dipanggil terpisah)
            p.value.d[i] -= self.lr * self.wd * p.value.d[i];
            p.value.d[i] -= lr * update;
        }
    }
}
