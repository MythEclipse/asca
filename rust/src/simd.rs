//! Kernel vektor AVX2/FMA untuk loop terdalam, tanpa dependensi luar.
//!
//! Deteksi fitur dilakukan sekali (runtime, bukan compile-time) supaya binary yang
//! sama tetap jalan di mesin tanpa AVX2 — di sana jalur skalarnya dipakai.
//!
//! Semua helper mempertahankan urutan penjumlahan per elemen output, jadi hasilnya
//! bit-identik dengan jalur skalar untuk bentuk yang sama; yang berubah hanya
//! seberapa banyak elemen yang diproses per instruksi.

use std::sync::OnceLock;

static AVX2: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn has_avx2() -> bool {
    *AVX2.get_or_init(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    })
}

/// `dst += a * src` untuk slice sepanjang sama.
#[inline]
pub fn axpy(dst: &mut [f32], a: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    if has_avx2() {
        unsafe { axpy_avx2(dst, a, src) };
    } else {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += a * s;
        }
    }
}

/// `sum(dst[i] * src[i])`.
#[inline]
pub fn dot(dst: &[f32], src: &[f32]) -> f32 {
    debug_assert_eq!(dst.len(), src.len());
    if has_avx2() {
        unsafe { dot_avx2(dst, src) }
    } else {
        let mut acc = 0.0f32;
        for (d, s) in dst.iter().zip(src.iter()) {
            acc += d * s;
        }
        acc
    }
}

/// `dst[i] += a * src[i] + b * src2[i]` — dipakai untuk dua akumulasi sekaligus
/// (mis. dW dan dx dalam satu lintasan baca).
#[inline]
pub fn axpy2(dst: &mut [f32], a: f32, src: &[f32], b: f32, src2: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(dst.len(), src2.len());
    if has_avx2() {
        unsafe { axpy2_avx2(dst, a, src, b, src2) };
    } else {
        for i in 0..dst.len() {
            dst[i] += a * src[i] + b * src2[i];
        }
    }
}

/// `dst *= a`.
#[inline]
pub fn scale(dst: &mut [f32], a: f32) {
    if has_avx2() {
        unsafe { scale_avx2(dst, a) };
    } else {
        for d in dst.iter_mut() {
            *d *= a;
        }
    }
}

/// `dst = a * dst + b * src` — satu lintasan untuk pembaruan state rekurensi.
#[inline]
pub fn scale_axpy(dst: &mut [f32], a: f32, b: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    if has_avx2() {
        unsafe { scale_axpy_avx2(dst, a, b, src) };
    } else {
        for i in 0..dst.len() {
            dst[i] = a * dst[i] + b * src[i];
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn scale_avx2(dst: &mut [f32], a: f32) {
    use std::arch::x86_64::*;
    unsafe {
    let n = dst.len();
    let mut i = 0;
    let av = _mm256_set1_ps(a);
    while i + 8 <= n {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(d, av));
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) *= a;
        i += 1;
    }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn scale_axpy_avx2(dst: &mut [f32], a: f32, b: f32, src: &[f32]) {
    use std::arch::x86_64::*;
    unsafe {
    let n = dst.len();
    let mut i = 0;
    let av = _mm256_set1_ps(a);
    let bv = _mm256_set1_ps(b);
    while i + 8 <= n {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i));
        let s = _mm256_loadu_ps(src.as_ptr().add(i));
        let r = _mm256_fmadd_ps(bv, s, _mm256_mul_ps(av, d));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = a * *dst.get_unchecked(i) + b * *src.get_unchecked(i);
        i += 1;
    }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(dst: &mut [f32], a: f32, src: &[f32]) {
    use std::arch::x86_64::*;
    unsafe {
    let n = dst.len();
    let mut i = 0;
    let av = _mm256_set1_ps(a);
    while i + 8 <= n {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i));
        let s = _mm256_loadu_ps(src.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(av, s, d));
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) += a * *src.get_unchecked(i);
        i += 1;
    }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(dst: &[f32], src: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
    let n = dst.len();
    let mut i = 0;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let a = _mm256_loadu_ps(dst.as_ptr().add(i));
        let b = _mm256_loadu_ps(src.as_ptr().add(i));
        acc = _mm256_fmadd_ps(a, b, acc);
        i += 8;
    }
    // horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let mut s = _mm_add_ps(hi, lo);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    let mut total = _mm_cvtss_f32(s);
    while i < n {
        total += *dst.get_unchecked(i) * *src.get_unchecked(i);
        i += 1;
    }
    total
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn axpy2_avx2(dst: &mut [f32], a: f32, src: &[f32], b: f32, src2: &[f32]) {
    use std::arch::x86_64::*;
    unsafe {
    let n = dst.len();
    let mut i = 0;
    let av = _mm256_set1_ps(a);
    let bv = _mm256_set1_ps(b);
    while i + 8 <= n {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i));
        let s = _mm256_loadu_ps(src.as_ptr().add(i));
        let s2 = _mm256_loadu_ps(src2.as_ptr().add(i));
        let mut acc = _mm256_fmadd_ps(av, s, d);
        acc = _mm256_fmadd_ps(bv, s2, acc);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), acc);
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) += a * *src.get_unchecked(i) + b * *src2.get_unchecked(i);
        i += 1;
    }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_matches_scalar() {
        for n in [0usize, 1, 7, 8, 9, 33, 128] {
            let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
            let mut a = vec![1.0f32; n];
            let mut b = vec![1.0f32; n];
            axpy(&mut a, 2.0, &src);
            for i in 0..n {
                b[i] += 2.0 * src[i];
            }
            for i in 0..n {
                assert!((a[i] - b[i]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn dot_matches_scalar() {
        for n in [0usize, 1, 7, 8, 9, 33, 128] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
            let mut want = 0.0f32;
            for i in 0..n {
                want += a[i] * b[i];
            }
            let got = dot(&a, &b);
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "n={n}");
        }
    }
}
