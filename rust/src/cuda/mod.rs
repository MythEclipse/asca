//! Jalur GPU opsional. Kalau tidak ada `nvcc` atau `libcuda`, semua fungsi di sini
//! melaporkan tidak tersedia dan model tetap berjalan di CPU.
//!
//! Untuk sekarang setiap panggilan mengalokasikan buffer dan menyalin bolak-balik.
//! Itu cukup untuk membuktikan paritas hasil dan mengukur batas atas percepatan,
//! tetapi bukan desain akhir: tensor harus tinggal di device supaya tidak ada
//! salinan per operasi.

pub mod sys;

use std::ffi::{CString, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tensor::Mat;

include!(concat!(env!("OUT_DIR"), "/gpu_build.rs"));

/// Berapa byte yang benar-benar menyeberangi PCIe, dan berapa kernel yang diluncurkan.
///
/// Ini bukan hiasan: keluhan "MoE forward sudah di GPU tapi masih makan 9%" punya satu
/// sebab yang bisa dihitung — `scatter_rows` mengunggah dan mengunduh seluruh matriks
/// keluaran untuk setiap expert. Tanpa penghitung ini, biaya transfer hanya bisa
/// ditaksir; dengan ini ia menjadi angka yang bisa dibandingkan sebelum/sesudah.
pub static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
pub static D2H_BYTES: AtomicU64 = AtomicU64::new(0);
pub static LAUNCHES: AtomicU64 = AtomicU64::new(0);

pub fn reset_stats() {
    H2D_BYTES.store(0, Ordering::Relaxed);
    D2H_BYTES.store(0, Ordering::Relaxed);
    LAUNCHES.store(0, Ordering::Relaxed);
}

/// Momen Adam (`m`, `v`) yang tinggal di VRAM, dikunci per parameter.
///
/// Kuncinya alamat buffer `m` di host: `Param` tidak pernah mengalokasi ulang `m`, jadi
/// alamatnya stabil sepanjang training. Cache dibersihkan lewat [`reset_param_cache`]
/// setiap kali model baru dibuat — tanpa itu, model yang kebetulan memakai alamat yang
/// sama akan mewarisi momen milik model sebelumnya, dan hasilnya salah tanpa error.
static PARAM_DEV: OnceLock<std::sync::Mutex<std::collections::HashMap<(usize, usize), (DevBuf, DevBuf)>>> =
    OnceLock::new();

fn param_dev() -> &'static std::sync::Mutex<std::collections::HashMap<(usize, usize), (DevBuf, DevBuf)>>
{
    PARAM_DEV.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Buang semua momen yang tersimpan di VRAM. Wajib dipanggil saat model baru dibuat.
pub fn reset_param_cache() {
    if let Some(m) = PARAM_DEV.get() {
        m.lock().unwrap().clear();
    }
}

/// Kunci cache untuk satu parameter.
pub fn param_key(m: &[f32]) -> (usize, usize) {
    (m.as_ptr() as usize, m.len())
}

/// Salinan momen dari VRAM, kalau parameter ini punya state di device.
/// Dipakai checkpoint resume: yang ditulis harus momen yang benar-benar dipakai,
/// bukan salinan host yang sudah basi.
pub fn param_moments(key: (usize, usize)) -> Option<(Vec<f32>, Vec<f32>)> {
    let map = param_dev().lock().unwrap();
    let (dm, dv) = map.get(&key)?;
    let n = key.1;
    let mut m = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    dm.download(&mut m).ok()?;
    dv.download(&mut v).ok()?;
    Some((m, v))
}

/// Ringkasan lalu lintas device. `steps` dipakai untuk membagi rata-rata per langkah.
pub fn stats_report(steps: u64) -> String {
    let h2d = H2D_BYTES.load(Ordering::Relaxed) as f64 / 1e6;
    let d2h = D2H_BYTES.load(Ordering::Relaxed) as f64 / 1e6;
    let l = LAUNCHES.load(Ordering::Relaxed);
    let s = steps.max(1) as f64;
    format!(
        "GPU lalu lintas: H2D {h2d:.1} MB, D2H {d2h:.1} MB, {l} kernel ({:.2} MB + {:.2} MB + {:.0} kernel per langkah)",
        h2d / s,
        d2h / s,
        l as f64 / s
    )
}

pub struct Gpu {
    pub name: String,
    pub sm_count: i32,
    module: sys::Module,
    /// Context disimpan supaya tidak di-destroy selama `Gpu` hidup.
    _ctx: sys::Context,
}

use std::collections::HashMap;
use std::sync::Mutex;

/// Caching memory allocator sederhana:
/// cuMemAlloc_v2 dan cuMemFree_v2 adalah operasi driver yang memblokir dan sangat lambat jika
/// dipanggil puluhan kali per langkah training. DevBuf sekarang meminjam dari pool per-ukuran,
/// dan saat di-drop dikembalikan ke pool untuk digunakan kembali tanpa pemanggilan driver.
static POOL: Mutex<Option<HashMap<usize, Vec<sys::DevicePtr>>>> = Mutex::new(None);

pub struct DevBuf(pub sys::DevicePtr, pub usize);

impl DevBuf {
    pub fn new(bytes: usize) -> Result<Self, String> {
        let sz = bytes.max(4).next_power_of_two();
        if let Ok(mut pool) = POOL.lock() {
            let map = pool.get_or_insert_with(HashMap::new);
            if let Some(list) = map.get_mut(&sz) {
                if let Some(p) = list.pop() {
                    return Ok(DevBuf(p, sz));
                }
            }
        }
        let mut p = sys::DevicePtr(0);
        let rc = unsafe { sys::cuMemAlloc_v2(&mut p, sz) };
        if rc != sys::SUCCESS {
            return Err(sys::err(rc));
        }
        Ok(DevBuf(p, sz))
    }
    pub fn upload(&self, data: &[f32]) -> Result<(), String> {
        H2D_BYTES.fetch_add(std::mem::size_of_val(data) as u64, Ordering::Relaxed);
        crate::prof::add_traffic(std::mem::size_of_val(data) as u64, false);
        let rc = unsafe {
            sys::cuMemcpyHtoD_v2(
                self.0,
                data.as_ptr() as *const c_void,
                std::mem::size_of_val(data),
            )
        };
        if rc != sys::SUCCESS {
            Err(sys::err(rc))
        } else {
            Ok(())
        }
    }
    pub fn download(&self, out: &mut [f32]) -> Result<(), String> {
        D2H_BYTES.fetch_add(std::mem::size_of_val(out) as u64, Ordering::Relaxed);
        crate::prof::add_traffic(std::mem::size_of_val(out) as u64, true);
        let rc = unsafe {
            sys::cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut c_void,
                self.0,
                std::mem::size_of_val(out),
            )
        };
        if rc != sys::SUCCESS {
            Err(sys::err(rc))
        } else {
            Ok(())
        }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if self.0.0 != 0 {
            if let Ok(mut pool) = POOL.lock() {
                if let Some(map) = pool.as_mut() {
                    map.entry(self.1).or_default().push(self.0);
                    return;
                }
            }
            unsafe {
                sys::cuMemFree_v2(self.0);
            }
        }
    }
}

impl Gpu {
    fn init() -> Option<Gpu> {
        // Setiap kegagalan dicatat BESERTA kode errornya. Versi sebelumnya hanya
        // `return None` di lima tempat berbeda, sehingga "GPU tidak tersedia" tidak bisa
        // dibedakan antara driver tidak ada, PTX gagal JIT, atau arsitektur tidak cocok.
        // Di mesin lain itu berarti menebak.
        macro_rules! gagal {
            ($tahap:expr, $rc:expr) => {{
                let _ = INIT_ERR.set(format!("{} (kode {})", $tahap, $rc));
                return None;
            }};
        }
        let ptx = match PTX {
            Some(p) => p,
            None => {
                let _ = INIT_ERR.set(format!("PTX tidak ter-embed saat build (arch {ARCH})"));
                return None;
            }
        };
        unsafe {
            let rc = sys::cuInit(0);
            if rc != sys::SUCCESS {
                gagal!("cuInit gagal - driver CUDA tidak termuat", rc);
            }
            let mut dev = sys::Device(0);
            let rc = sys::cuDeviceGet(&mut dev, 0);
            if rc != sys::SUCCESS {
                gagal!("cuDeviceGet gagal - tidak ada GPU terlihat", rc);
            }
            let mut ctx = sys::Context(std::ptr::null_mut());
            let rc = sys::cuCtxCreate_v2(&mut ctx, 0, dev);
            if rc != sys::SUCCESS {
                gagal!("cuCtxCreate gagal", rc);
            }
            let mut buf = vec![0u8; 256];
            sys::cuDeviceGetName(buf.as_mut_ptr() as *mut i8, 255, dev);
            let name = CString::new(buf.split(|&b| b == 0).next().unwrap_or(&buf))
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "unknown".to_string());
            let mut sm_count = 0i32;
            sys::cuDeviceGetAttribute(&mut sm_count, sys::ATTRIBUTE_MULTIPROCESSORS, dev);

            let mut module = sys::Module(std::ptr::null_mut());
            let src = match CString::new(ptx) {
                Ok(s) => s,
                Err(_) => {
                    let _ = INIT_ERR.set("PTX mengandung byte nol".to_string());
                    return None;
                }
            };
            let rc = sys::cuModuleLoadData(&mut module, src.as_ptr() as *const c_void);
            if rc != sys::SUCCESS {
                gagal!("PTX gagal dimuat/JIT untuk arch target", rc);
            }
            Some(Gpu {
                name,
                sm_count,
                module,
                _ctx: ctx,
            })
        }
    }

    fn func(&self, name: &str) -> Result<sys::Function, String> {
        let cname = CString::new(name).map_err(|e| e.to_string())?;
        let mut f = sys::Function(std::ptr::null_mut());
        let rc = unsafe { sys::cuModuleGetFunction(&mut f, self.module, cname.as_ptr()) };
        if rc != sys::SUCCESS {
            return Err(format!("kernel {name} tidak ditemukan: {}", sys::err(rc)));
        }
        Ok(f)
    }

    fn launch(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        mut args: Vec<*mut c_void>,
    ) -> Result<(), String> {
        let f = self.func(name)?;
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        if shared > 48 * 1024 {
            unsafe {
                sys::cuFuncSetAttribute(f, sys::FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED, shared as i32);
            }
        }
        let rc = unsafe {
            sys::cuLaunchKernel(
                f,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if rc != sys::SUCCESS {
            return Err(sys::err(rc));
        }
        Ok(())
    }

    /// C[M,N] = A[M,K] @ B[K,N]
    pub fn matmul(&self, a: &Mat, b: &Mat) -> Result<Mat, String> {
        assert_eq!(a.cols, b.rows, "matmul: dimensi tidak cocok");
        let (m, k, n) = (a.rows, a.cols, b.cols);
        let da = DevBuf::new(a.d.len() * 4)?;
        let db = DevBuf::new(b.d.len() * 4)?;
        let dc = DevBuf::new(m * n * 4)?;
        da.upload(&a.d)?;
        db.upload(&b.d)?;
        let (mut mi, mut ki, mut ni) = (m as i32, k as i32, n as i32);
        self.launch(
            "matmul_f32",
            (((n as u32).div_ceil(32)), ((m as u32).div_ceil(32)), 1),
            (32, 32, 1),
            0,
            vec![
                &da.0 as *const _ as *mut c_void,
                &db.0 as *const _ as *mut c_void,
                &dc.0 as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        let mut out = vec![0.0f32; m * n];
        dc.download(&mut out)?;
        Ok(Mat::from_vec(m, n, out))
    }

    /// C[M,N] = A[K,M]^T @ B[K,N]
    pub fn matmul_ta(&self, a: &Mat, b: &Mat) -> Result<Mat, String> {
        let (k, m) = (a.rows, a.cols);
        assert_eq!(k, b.rows, "matmul_ta: dimensi tidak cocok");
        let n = b.cols;
        let da = DevBuf::new(a.d.len() * 4)?;
        let db = DevBuf::new(b.d.len() * 4)?;
        let dc = DevBuf::new(m * n * 4)?;
        da.upload(&a.d)?;
        db.upload(&b.d)?;
        let (mut mi, mut ki, mut ni) = (m as i32, k as i32, n as i32);
        self.launch(
            "matmul_ta_f32",
            (((n as u32).div_ceil(32)), ((m as u32).div_ceil(32)), 1),
            (32, 32, 1),
            0,
            vec![
                &da.0 as *const _ as *mut c_void,
                &db.0 as *const _ as *mut c_void,
                &dc.0 as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        let mut out = vec![0.0f32; m * n];
        dc.download(&mut out)?;
        Ok(Mat::from_vec(m, n, out))
    }

    /// `C = A @ B` dengan semua operand sudah di device — tanpa satu pun transfer.
    fn matmul_raw(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        let (mut mi, mut ki, mut ni) = (m as i32, k as i32, n as i32);
        self.launch(
            "matmul_f32",
            (((n as u32).div_ceil(32)), ((m as u32).div_ceil(32)), 1),
            (32, 32, 1),
            0,
            vec![
                &a.0 as *const _ as *mut c_void,
                &b.0 as *const _ as *mut c_void,
                &c.0 as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
            ],
        )
    }

    /// `C = A^T @ B` dengan semua operand sudah di device.
    fn matmul_ta_raw(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        k: usize,
        m: usize,
        n: usize,
    ) -> Result<(), String> {
        let (mut mi, mut ki, mut ni) = (m as i32, k as i32, n as i32);
        self.launch(
            "matmul_ta_f32",
            (((n as u32).div_ceil(32)), ((m as u32).div_ceil(32)), 1),
            (32, 32, 1),
            0,
            vec![
                &a.0 as *const _ as *mut c_void,
                &b.0 as *const _ as *mut c_void,
                &c.0 as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
            ],
        )
    }

    /// Gather baris di device: `dst[r] = src[idx[r]]`, tanpa transfer.
    fn gather_raw(
        &self,
        src: &DevBuf,
        idx: &DevBuf,
        n: usize,
        cols: usize,
        dst: &DevBuf,
    ) -> Result<(), String> {
        let (mut nr, mut c) = (n as i32, cols as i32);
        self.launch(
            "gather_rows_f32",
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &src.0 as *const _ as *mut c_void,
                &idx.0 as *const _ as *mut c_void,
                &dst.0 as *const _ as *mut c_void,
                &mut nr as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
            ],
        )
    }

    /// Scatter baris di device: `dst[idx[r]] = src[r]`, tanpa transfer.
    fn scatter_raw(
        &self,
        src: &DevBuf,
        idx: &DevBuf,
        n: usize,
        cols: usize,
        dst: &DevBuf,
    ) -> Result<(), String> {
        let (mut nr, mut c) = (n as i32, cols as i32);
        self.launch(
            "scatter_rows_f32",
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &src.0 as *const _ as *mut c_void,
                &idx.0 as *const _ as *mut c_void,
                &dst.0 as *const _ as *mut c_void,
                &mut nr as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
            ],
        )
    }


    /// Tambah bias kolom di device mulai `off` elemen (`X[i] += b[off + i % cols]`).
    fn add_bias_raw(
        &self,
        buf: &DevBuf,
        bias: &DevBuf,
        off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        let (mut r, mut c) = (rows as i32, cols as i32);
        let biased = sys::DevicePtr(bias.0.0 + (off * 4) as u64);
        self.launch(
            "add_bias_f32",
            (((rows * cols) as u32).div_ceil(256), 1, 1),
            (256, 1, 1),
            0,
            vec![
                &buf.0 as *const _ as *mut c_void,
                &biased as *const _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
            ],
        )
    }

    /// GEMM `C = A @ B[:, offb:offb+N]` dengan `B` berstride `ldb`, semua di device.
    #[allow(clippy::too_many_arguments)]
    fn matmul_ld_raw(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        m: usize,
        k: usize,
        n: usize,
        ldb: usize,
        offb: usize,
        b_off_elems: usize,
    ) -> Result<(), String> {
        let (mut mi, mut ki, mut ni) = (m as i32, k as i32, n as i32);
        let (mut ldb_i, mut offb_i) = (ldb as i32, offb as i32);
        let bp = sys::DevicePtr(b.0.0 + (b_off_elems * 4) as u64);
        self.launch(
            "matmul_ld_f32",
            (((n as u32).div_ceil(32)), ((m as u32).div_ceil(32)), 1),
            (32, 32, 1),
            0,
            vec![
                &a.0 as *const _ as *mut c_void,
                &bp as *const _ as *mut c_void,
                &c.0 as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
                &mut ldb_i as *mut _ as *mut c_void,
                &mut offb_i as *mut _ as *mut c_void,
            ],
        )
    }

    /// MoE forward dengan seluruh intermediate tetap di VRAM.
    ///
    /// Versi per operasi mengunggah `xn` untuk tiap expert, mengunduh keluarannya,
    /// mengunggah lagi untuk GEMM kedua, lalu scatter dengan menyalin seluruh matriks
    /// keluaran naik-turun. Terukur **382 MB per langkah** hanya untuk MoE forward — 49%
    /// dari seluruh lalu lintas device. Di sini `xn`, `w1`, `w2`, `b1`, `b2` diunggah
    /// sekali per layer; gather, GEMM, bias, SiLU, dan scatter berjalan di device.
    ///
    /// `w1` dipakai bersama semua expert lewat `matmul_ld_f32` (offset kolom), jadi tidak
    /// ada lagi salinan kolom per expert di host. Yang turun hanya `out` dan `pre`
    /// (`hidden` dihitung ulang di host dengan rumus yang sama persis dengan kernel).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_forward_dev(
        &self,
        xn: &Mat,
        w1: &Mat,
        b1: &[f32],
        w2: &Mat,
        b2: &[f32],
        buckets: &[Vec<usize>],
        d_ff: usize,
    ) -> Result<(Mat, MoeDev), String> {
        let (t_len, d) = (xn.rows, xn.cols);
        let w1_cols = w1.cols;
        let dxn = DevBuf::new(t_len * d * 4)?;
        let dw1 = DevBuf::new(w1.d.len() * 4)?;
        let dw2 = DevBuf::new(w2.d.len() * 4)?;
        let db1 = DevBuf::new(b1.len() * 4)?;
        let db2 = DevBuf::new(b2.len() * 4)?;
        let dout = DevBuf::new(t_len * d * 4)?;
        dxn.upload(&xn.d)?;
        dw1.upload(&w1.d)?;
        dw2.upload(&w2.d)?;
        db1.upload(b1)?;
        db2.upload(b2)?;
        let rc = unsafe { sys::cuMemsetD32_v2(dout.0, 0, t_len * d) };
        if rc != sys::SUCCESS {
            return Err(sys::err(rc));
        }
        let mut experts: Vec<MoeExpertDev> = Vec::with_capacity(buckets.len());
        for (e, rows) in buckets.iter().enumerate() {
            if rows.is_empty() {
                let empty = DevBuf::new(4)?;
                experts.push(MoeExpertDev {
                    idx: empty,
                    pre: DevBuf::new(4)?,
                    hidden: DevBuf::new(4)?,
                    n: 0,
                });
                continue;
            }
            let n = rows.len();
            let idx_i: Vec<i32> = rows.iter().map(|&r| r as i32).collect();
            let didx = DevBuf::new(n * 4)?;
            unsafe {
                let rc = sys::cuMemcpyHtoD_v2(didx.0, idx_i.as_ptr() as *const c_void, n * 4);
                if rc != sys::SUCCESS {
                    return Err(sys::err(rc));
                }
            }
            let dxe = DevBuf::new(n * d * 4)?;
            let dpe = DevBuf::new(n * d_ff * 4)?;
            let dhe = DevBuf::new(n * d_ff * 4)?;
            let doe = DevBuf::new(n * d * 4)?;
            self.gather_raw(&dxn, &didx, n, d, &dxe)?;
            // pe = xe @ w1[:, e*d_ff : (e+1)*d_ff] + b1[e*d_ff..]
            self.matmul_ld_raw(&dxe, &dw1, &dpe, n, d, d_ff, w1_cols, e * d_ff, 0)?;
            self.add_bias_raw(&dpe, &db1, e * d_ff, n, d_ff)?;
            // he = silu(pe) ke buffer terpisah: `pre` dibutuhkan utuh di backward
            self.silu_out_raw(&dhe, &dpe, n * d_ff)?;
            // oe = he @ w2[e*d_ff.., :] + b2
            self.matmul_ld_raw(&dhe, &dw2, &doe, n, d_ff, d, d, 0, e * d_ff * d)?;
            self.add_bias_raw(&doe, &db2, 0, n, d)?;
            self.scatter_raw(&doe, &didx, n, d, &dout)?;
            experts.push(MoeExpertDev {
                idx: didx,
                pre: dpe,
                hidden: dhe,
                n,
            });
        }
        unsafe { sys::cuCtxSynchronize() };
        let mut out = vec![0.0f32; t_len * d];
        dout.download(&mut out)?;
        Ok((
            Mat::from_vec(t_len, d, out),
            MoeDev {
                xn: dxn,
                w1: dw1,
                w2: dw2,
                experts,
                t_len,
                d,
                d_ff,
            },
        ))
    }

    /// SiLU ke buffer lain di device.
    fn silu_out_raw(&self, dst: &DevBuf, src: &DevBuf, n: usize) -> Result<(), String> {
        let mut n_i = n as i32;
        self.launch(
            "silu_out_f32",
            ((n as u32).div_ceil(256), 1, 1),
            (256, 1, 1),
            0,
            vec![
                &dst.0 as *const _ as *mut c_void,
                &src.0 as *const _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
            ],
        )
    }

    /// Turunan SiLU di tempat.
    fn silu_bwd_raw(&self, dpre: &DevBuf, pre: &DevBuf, n: usize) -> Result<(), String> {
        let mut n_i = n as i32;
        self.launch(
            "silu_bwd_f32",
            ((n as u32).div_ceil(256), 1, 1),
            (256, 1, 1),
            0,
            vec![
                &dpre.0 as *const _ as *mut c_void,
                &pre.0 as *const _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
            ],
        )
    }

    /// Jumlah kolom; `acc` menumpuk ke `dst` (dipakai antar expert).
    fn colsum_raw(
        &self,
        src: &DevBuf,
        dst: &DevBuf,
        dst_off: usize,
        rows: usize,
        cols: usize,
        acc: bool,
    ) -> Result<(), String> {
        let (mut r, mut c) = (rows as i32, cols as i32);
        let mut a = i32::from(acc);
        let dp = sys::DevicePtr(dst.0.0 + (dst_off * 4) as u64);
        self.launch(
            "colsum_f32",
            ((cols as u32).div_ceil(128), 1, 1),
            (128, 1, 1),
            0,
            vec![
                &src.0 as *const _ as *mut c_void,
                &dp as *const _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut a as *mut _ as *mut c_void,
            ],
        )
    }

    /// `C[M,N] = A[K,M]^T @ B[K,N]`, ditulis ke kolom `offc` dari C berstride `ldc`.
    #[allow(clippy::too_many_arguments)]
    fn matmul_ta_ld_raw(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        k: usize,
        m: usize,
        n: usize,
        ldc: usize,
        offc: usize,
        c_off_elems: usize,
    ) -> Result<(), String> {
        let (mut mi, mut ki, mut ni) = (m as i32, k as i32, n as i32);
        let (mut ldc_i, mut offc_i) = (ldc as i32, offc as i32);
        let cp = sys::DevicePtr(c.0.0 + (c_off_elems * 4) as u64);
        self.launch(
            "matmul_ta_ld_f32",
            (((n as u32).div_ceil(32)), ((m as u32).div_ceil(32)), 1),
            (32, 32, 1),
            0,
            vec![
                &a.0 as *const _ as *mut c_void,
                &b.0 as *const _ as *mut c_void,
                &cp as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
                &mut ldc_i as *mut _ as *mut c_void,
                &mut offc_i as *mut _ as *mut c_void,
            ],
        )
    }

    /// `C[M,N] = A[M,K] @ B[:, offb:offb+N]^T`, B berstride `ldb` baris.
    #[allow(clippy::too_many_arguments)]
    fn matmul_tb_raw(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        m: usize,
        k: usize,
        n: usize,
        ldb: usize,
        offb: usize,
        b_off_elems: usize,
    ) -> Result<(), String> {
        let (mut mi, mut ki, mut ni) = (m as i32, k as i32, n as i32);
        let (mut ldb_i, mut offb_i) = (ldb as i32, offb as i32);
        let bp = sys::DevicePtr(b.0.0 + (b_off_elems * 4) as u64);
        self.launch(
            "matmul_tb_f32",
            (((n as u32).div_ceil(32)), ((m as u32).div_ceil(32)), 1),
            (32, 32, 1),
            0,
            vec![
                &a.0 as *const _ as *mut c_void,
                &bp as *const _ as *mut c_void,
                &c.0 as *const _ as *mut c_void,
                &mut mi as *mut _ as *mut c_void,
                &mut ki as *mut _ as *mut c_void,
                &mut ni as *mut _ as *mut c_void,
                &mut ldb_i as *mut _ as *mut c_void,
                &mut offb_i as *mut _ as *mut c_void,
            ],
        )
    }

    /// MoE backward seluruhnya di device.
    ///
    /// Per expert: gather dout -> dW2 = hidden^T dout -> db2 = jumlah dout ->
    /// dpre = dout w2^T -> kali turunan SiLU -> db1 -> dW1 = xn^T dpre ->
    /// dxn = dpre w1^T -> scatter. `pre`/`hidden`/bobot sudah ada di VRAM dari forward,
    /// jadi tidak ada satu pun yang perlu naik lagi.
    pub fn moe_bwd_dev(&self, dev: &MoeDev, dout: &Mat) -> Result<MoeBwd, String> {
        let (t_len, d, d_ff) = (dev.t_len, dev.d, dev.d_ff);
        let n_experts = dev.experts.len();
        let w1_cols = d_ff * n_experts;
        let ddout = DevBuf::new(t_len * d * 4)?;
        ddout.upload(&dout.d)?;
        let ddw1 = DevBuf::new(d * w1_cols * 4)?;
        let ddw2 = DevBuf::new(n_experts * d_ff * d * 4)?;
        let ddb1 = DevBuf::new(n_experts * d_ff * 4)?;
        let ddb2 = DevBuf::new(d * 4)?;
        let ddxn = DevBuf::new(t_len * d * 4)?;
        for buf in [&ddw1, &ddw2, &ddb1, &ddb2, &ddxn] {
            let rc = unsafe { sys::cuMemsetD32_v2(buf.0, 0, buf.1 / 4) };
            if rc != sys::SUCCESS {
                return Err(sys::err(rc));
            }
        }
        for (e, ex) in dev.experts.iter().enumerate() {
            let n = ex.n;
            if n == 0 {
                continue;
            }
            let ddhe = DevBuf::new(n * d * 4)?;
            self.gather_raw(&ddout, &ex.idx, n, d, &ddhe)?;
            // dW2_e = hidden^T dhe -> baris e*d_ff .. (e+1)*d_ff
            self.matmul_ta_ld_raw(
                &ex.hidden,
                &ddhe,
                &ddw2,
                n,
                d_ff,
                d,
                d,
                0,
                e * d_ff * d,
            )?;
            self.colsum_raw(&ddhe, &ddb2, 0, n, d, true)?;
            // dpre = dhe @ w2[e*d_ff..]^T lalu dikali turunan SiLU
            let ddpre = DevBuf::new(n * d_ff * 4)?;
            self.matmul_tb_raw(&ddhe, &dev.w2, &ddpre, n, d, d_ff, d, 0, e * d_ff * d)?;
            self.silu_bwd_raw(&ddpre, &ex.pre, n * d_ff)?;
            self.colsum_raw(&ddpre, &ddb1, e * d_ff, n, d_ff, true)?;
            // dW1_e = xn^T dpre -> kolom e*d_ff
            let dxe = DevBuf::new(n * d * 4)?;
            self.gather_raw(&dev.xn, &ex.idx, n, d, &dxe)?;
            self.matmul_ta_ld_raw(&dxe, &ddpre, &ddw1, n, d, d_ff, w1_cols, e * d_ff, 0)?;
            // dxn_e = dpre @ w1[:, e*d_ff..]^T
            let ddxn_e = DevBuf::new(n * d * 4)?;
            self.matmul_tb_raw(
                &ddpre,
                &dev.w1,
                &ddxn_e,
                n,
                d_ff,
                d,
                w1_cols,
                e * d_ff,
                0,
            )?;
            self.scatter_raw(&ddxn_e, &ex.idx, n, d, &ddxn)?;
        }
        unsafe { sys::cuCtxSynchronize() };
        let mut w1v = vec![0.0f32; d * w1_cols];
        let mut w2v = vec![0.0f32; n_experts * d_ff * d];
        let mut b1v = vec![0.0f32; n_experts * d_ff];
        let mut b2v = vec![0.0f32; d];
        let mut xnv = vec![0.0f32; t_len * d];
        ddw1.download(&mut w1v)?;
        ddw2.download(&mut w2v)?;
        ddb1.download(&mut b1v)?;
        ddb2.download(&mut b2v)?;
        ddxn.download(&mut xnv)?;
        Ok(MoeBwd {
            d_w1: Mat::from_vec(d, w1_cols, w1v),
            d_b1: b1v,
            d_w2: Mat::from_vec(n_experts * d_ff, d, w2v),
            d_b2: b2v,
            dxn: Mat::from_vec(t_len, d, xnv),
        })
    }

    /// `h_final @ head` **tanpa mengunduh hasilnya**. Matriks (T x vocab) — 100 MB pada
    /// config 12,7M — tetap di VRAM dan langsung dipakai cross-entropy backward.
    pub fn head_forward_dev(&self, h_final: &Mat, head: &Mat) -> Result<LogitsDev, String> {
        let (t_len, d) = (h_final.rows, h_final.cols);
        let v = head.cols;
        assert_eq!(head.rows, d, "head_forward_dev: dimensi head tidak cocok");
        let d_h = DevBuf::new(h_final.d.len() * 4)?;
        let d_w = DevBuf::new(head.d.len() * 4)?;
        let d_l = DevBuf::new(t_len * v * 4)?;
        d_h.upload(&h_final.d)?;
        d_w.upload(&head.d)?;
        self.matmul_raw(&d_h, &d_w, &d_l, t_len, d, v)?;
        unsafe { sys::cuCtxSynchronize() };
        Ok(LogitsDev {
            logits: d_l,
            h_final: d_h,
            t_len,
            vocab: v,
        })
    }

    /// Cross-entropy backward + dua GEMM kepala, seluruhnya di device.
    ///
    /// Mengembalikan `(loss, d_head, dh)`. `dlogits` — matriks (T x vocab) yang dulu
    /// diunduh lalu diunggah dua kali lagi — tidak pernah menyentuh host.
    pub fn head_bwd_dev(
        &self,
        dev: &LogitsDev,
        targets: &[u32],
        head_t: &Mat,
    ) -> Result<(f32, Mat, Mat), String> {
        let (t_len, v) = (dev.t_len, dev.vocab);
        assert_eq!(targets.len(), t_len, "head_bwd_dev: targets tidak cocok");
        let d = head_t.cols;
        assert_eq!(head_t.rows, v, "head_bwd_dev: head_t harus (vocab, d)");
        let d_targets = DevBuf::new(t_len * 4)?;
        let d_dlogits = DevBuf::new(t_len * v * 4)?;
        let d_losses = DevBuf::new(t_len * 4)?;
        let d_head = DevBuf::new(d * v * 4)?;
        let d_dh = DevBuf::new(t_len * d * 4)?;
        let d_ht = DevBuf::new(head_t.d.len() * 4)?;
        d_ht.upload(&head_t.d)?;
        unsafe {
            let rc =
                sys::cuMemcpyHtoD_v2(d_targets.0, targets.as_ptr() as *const c_void, t_len * 4);
            if rc != sys::SUCCESS {
                return Err(sys::err(rc));
            }
        }
        let (mut t_i, mut v_i) = (t_len as i32, v as i32);
        let mut inv_t = 1.0f32 / t_len as f32;
        self.launch(
            "cross_entropy_bwd_f32",
            (t_len as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &dev.logits.0 as *const _ as *mut c_void,
                &d_targets.0 as *const _ as *mut c_void,
                &d_dlogits.0 as *const _ as *mut c_void,
                &d_losses.0 as *const _ as *mut c_void,
                &mut t_i as *mut _ as *mut c_void,
                &mut v_i as *mut _ as *mut c_void,
                &mut inv_t as *mut _ as *mut c_void,
            ],
        )?;
        // d_head = h_final^T @ dlogits ; dh = dlogits @ head^T
        self.matmul_ta_raw(&dev.h_final, &d_dlogits, &d_head, t_len, d, v)?;
        self.matmul_raw(&d_dlogits, &d_ht, &d_dh, t_len, v, d)?;
        unsafe { sys::cuCtxSynchronize() };
        let mut losses = vec![0.0f32; t_len];
        d_losses.download(&mut losses)?;
        let loss = losses.iter().sum::<f32>() * inv_t;
        let mut d_head_out = vec![0.0f32; d * v];
        d_head.download(&mut d_head_out)?;
        let mut d_h_out = vec![0.0f32; t_len * d];
        d_dh.download(&mut d_h_out)?;
        Ok((
            loss,
            Mat::from_vec(d, v, d_head_out),
            Mat::from_vec(t_len, d, d_h_out),
        ))
    }

    /// RMSNorm per baris di GPU. Mengembalikan (y, inv_rms).
    pub fn rmsnorm(&self, x: &Mat, eps: f32) -> Result<(Mat, Vec<f32>), String> {
        let (rows, cols) = (x.rows, x.cols);
        let dx = DevBuf::new(x.d.len() * 4)?;
        let dy = DevBuf::new(x.d.len() * 4)?;
        let dinv = DevBuf::new(rows * 4)?;
        dx.upload(&x.d)?;
        let (mut r, mut c) = (rows as i32, cols as i32);
        let mut e = eps;
        self.launch(
            "rmsnorm_f32",
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &dx.0 as *const _ as *mut c_void,
                &dy.0 as *const _ as *mut c_void,
                &dinv.0 as *const _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut e as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        let mut out = vec![0.0f32; x.d.len()];
        dy.download(&mut out)?;
        let mut inv = vec![0.0f32; rows];
        dinv.download(&mut inv)?;
        Ok((Mat::from_vec(rows, cols, out), inv))
    }

    /// Delta rule forward untuk semua head sekaligus (satu blok per head).
    #[allow(clippy::too_many_arguments)]
    /// Cross-entropy loss dan backward di GPU.
    /// Mengembalikan loss rata-rata dan mengisi `dlogits`.
    pub fn cross_entropy_bwd(&self, logits: &Mat, targets: &[u32]) -> Result<(f32, Mat), String> {
        let (t_len, v) = (logits.rows, logits.cols);
        assert_eq!(
            t_len,
            targets.len(),
            "targets len harus sama dengan logits rows"
        );
        let d_logits = DevBuf::new(t_len * v * 4)?;
        let d_targets = DevBuf::new(t_len * 4)?;
        let d_dlogits = DevBuf::new(t_len * v * 4)?;
        let d_losses = DevBuf::new(t_len * 4)?;

        d_logits.upload(&logits.d)?;
        unsafe {
            let rc =
                sys::cuMemcpyHtoD_v2(d_targets.0, targets.as_ptr() as *const c_void, t_len * 4);
            if rc != sys::SUCCESS {
                return Err(sys::err(rc));
            }
        }

        let mut t_i = t_len as i32;
        let mut v_i = v as i32;
        let mut inv_t = 1.0f32 / t_len as f32;

        self.launch(
            "cross_entropy_bwd_f32",
            (t_len as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &d_logits.0 as *const _ as *mut c_void,
                &d_targets.0 as *const _ as *mut c_void,
                &d_dlogits.0 as *const _ as *mut c_void,
                &d_losses.0 as *const _ as *mut c_void,
                &mut t_i as *mut _ as *mut c_void,
                &mut v_i as *mut _ as *mut c_void,
                &mut inv_t as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };

        let mut losses = vec![0.0f32; t_len];
        d_losses.download(&mut losses)?;
        let loss = losses.iter().sum::<f32>() * inv_t;

        let mut dlogits_out = vec![0.0f32; t_len * v];
        d_dlogits.download(&mut dlogits_out)?;

        Ok((loss, Mat::from_vec(t_len, v, dlogits_out)))
    }

    /// AdamW dengan momen `m`/`v` yang tetap di VRAM antar langkah.
    ///
    /// Versi per operasi menyalin 7 buffer per parameter per langkah (nilai, gradien, m,
    /// v naik; nilai, m, v turun) — 357 MB/langkah pada model 12,7M. Dengan momen yang
    /// tinggal di device, tinggal 3 salinan: nilai + gradien naik, nilai turun.
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_resident(
        &self,
        key: (usize, usize),
        val: &mut [f32],
        grad: &[f32],
        m0: &[f32],
        v0: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps_t: f32,
        wd: f32,
    ) -> Result<(), String> {
        let n = val.len();
        assert_eq!(n, grad.len());
        let d_val = DevBuf::new(n * 4)?;
        let d_grad = DevBuf::new(n * 4)?;
        d_val.upload(val)?;
        d_grad.upload(grad)?;
        let mut map = param_dev().lock().unwrap();
        if !map.contains_key(&key) {
            let dm = DevBuf::new(n * 4)?;
            let dv = DevBuf::new(n * 4)?;
            dm.upload(m0)?;
            dv.upload(v0)?;
            map.insert(key, (dm, dv));
        }
        let (dm, dv) = map.get(&key).expect("momen baru saja disisipkan");
        let mut n_i = n as i32;
        let (mut lr_c, mut b1, mut b2, mut eps_c, mut wd_c) = (lr, beta1, beta2, eps_t, wd);
        self.launch(
            "adamw_f32",
            (n.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &d_val.0 as *const _ as *mut c_void,
                &d_grad.0 as *const _ as *mut c_void,
                &dm.0 as *const _ as *mut c_void,
                &dv.0 as *const _ as *mut c_void,
                &mut lr_c as *mut _ as *mut c_void,
                &mut b1 as *mut _ as *mut c_void,
                &mut b2 as *mut _ as *mut c_void,
                &mut eps_c as *mut _ as *mut c_void,
                &mut wd_c as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        d_val.download(val)?;
        Ok(())
    }

    /// Update AdamW di GPU untuk satu parameter.
    pub fn adamw_update(
        &self,
        val: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps_t: f32,
        wd: f32,
    ) -> Result<(), String> {
        let n = val.len();
        assert_eq!(n, grad.len());
        assert_eq!(n, m.len());
        assert_eq!(n, v.len());

        let d_val = DevBuf::new(n * 4)?;
        let d_grad = DevBuf::new(n * 4)?;
        let d_m = DevBuf::new(n * 4)?;
        let d_v = DevBuf::new(n * 4)?;

        d_val.upload(val)?;
        d_grad.upload(grad)?;
        d_m.upload(m)?;
        d_v.upload(v)?;

        let mut lr_i = lr;
        let mut b1_i = beta1;
        let mut b2_i = beta2;
        let mut eps_i = eps_t;
        let mut wd_i = wd;
        let mut n_i = n as i32;

        let block = 256u32;
        let grid = (n as u32).div_ceil(block);

        self.launch(
            "adamw_f32",
            (grid, 1, 1),
            (block, 1, 1),
            0,
            vec![
                &d_val.0 as *const _ as *mut c_void,
                &d_grad.0 as *const _ as *mut c_void,
                &d_m.0 as *const _ as *mut c_void,
                &d_v.0 as *const _ as *mut c_void,
                &mut lr_i as *mut _ as *mut c_void,
                &mut b1_i as *mut _ as *mut c_void,
                &mut b2_i as *mut _ as *mut c_void,
                &mut eps_i as *mut _ as *mut c_void,
                &mut wd_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };

        d_val.download(val)?;
        d_m.download(m)?;
        d_v.download(v)?;

        Ok(())
    }

    /// RMSNorm backward di GPU: dx = inv * (dy - xhat * mean(dy * xhat)).
    pub fn rmsnorm_bwd(&self, x: &Mat, dy: &Mat, eps: f32) -> Result<Mat, String> {
        let (rows, cols) = (x.rows, x.cols);
        assert_eq!(rows, dy.rows);
        assert_eq!(cols, dy.cols);
        let dx_buf = DevBuf::new(x.d.len() * 4)?;
        let dy_buf = DevBuf::new(dy.d.len() * 4)?;
        let dout_buf = DevBuf::new(x.d.len() * 4)?;

        dx_buf.upload(&x.d)?;
        dy_buf.upload(&dy.d)?;

        let (mut r, mut c) = (rows as i32, cols as i32);
        let mut e = eps;
        self.launch(
            "rmsnorm_bwd_f32",
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &dx_buf.0 as *const _ as *mut c_void,
                &dy_buf.0 as *const _ as *mut c_void,
                &dout_buf.0 as *const _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut e as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        let mut out = vec![0.0f32; x.d.len()];
        dout_buf.download(&mut out)?;
        Ok(Mat::from_vec(rows, cols, out))
    }

    /// Kumpulkan baris `idx` dari `src` menjadi matriks kontigu (n x cols).
    pub fn gather_rows(&self, src: &Mat, idx: &[usize]) -> Result<Mat, String> {
        let (n, cols) = (idx.len(), src.cols);
        let dsrc = DevBuf::new(src.d.len() * 4)?;
        let didx = DevBuf::new(n * 4)?;
        let ddst = DevBuf::new(n * cols * 4)?;
        dsrc.upload(&src.d)?;
        let idx_i: Vec<i32> = idx.iter().map(|&i| i as i32).collect();
        unsafe {
            let rc = sys::cuMemcpyHtoD_v2(didx.0, idx_i.as_ptr() as *const c_void, n * 4);
            if rc != sys::SUCCESS {
                return Err(sys::err(rc));
            }
        }
        let (mut nr, mut c) = (n as i32, cols as i32);
        self.launch(
            "gather_rows_f32",
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &dsrc.0 as *const _ as *mut c_void,
                &didx.0 as *const _ as *mut c_void,
                &ddst.0 as *const _ as *mut c_void,
                &mut nr as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        let mut out = vec![0.0f32; n * cols];
        ddst.download(&mut out)?;
        Ok(Mat::from_vec(n, cols, out))
    }

    /// Sebar baris `src` ke posisi `idx` pada `dst` (dst diubah di tempat).
    pub fn scatter_rows(&self, src: &Mat, idx: &[usize], dst: &mut Mat) -> Result<(), String> {
        let (n, cols) = (idx.len(), src.cols);
        let dsrc = DevBuf::new(src.d.len() * 4)?;
        let didx = DevBuf::new(n * 4)?;
        let ddst = DevBuf::new(dst.d.len() * 4)?;
        dsrc.upload(&src.d)?;
        ddst.upload(&dst.d)?;
        let idx_i: Vec<i32> = idx.iter().map(|&i| i as i32).collect();
        unsafe {
            let rc = sys::cuMemcpyHtoD_v2(didx.0, idx_i.as_ptr() as *const c_void, n * 4);
            if rc != sys::SUCCESS {
                return Err(sys::err(rc));
            }
        }
        let (mut nr, mut c) = (n as i32, cols as i32);
        self.launch(
            "scatter_rows_f32",
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            vec![
                &dsrc.0 as *const _ as *mut c_void,
                &didx.0 as *const _ as *mut c_void,
                &ddst.0 as *const _ as *mut c_void,
                &mut nr as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        ddst.download(&mut dst.d)?;
        Ok(())
    }

    /// SiLU di tempat.
    pub fn silu_inplace(&self, x: &mut Mat) -> Result<(), String> {
        let n = x.d.len();
        let dx = DevBuf::new(n * 4)?;
        dx.upload(&x.d)?;
        let mut n_i = n as i32;
        self.launch(
            "silu_f32",
            ((n as u32).div_ceil(256), 1, 1),
            (256, 1, 1),
            0,
            vec![
                &dx.0 as *const _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        dx.download(&mut x.d)?;
        Ok(())
    }

    /// Delta rule forward untuk seluruh (head, sekuens) sekaligus, dengan state yang
    /// **tetap di device**.
    ///
    /// Satu blok per (head, sekuens), jadi `batch` sekuens yang ditumpuk tidak saling
    /// bocor. `q`, `k`, `v`, `alpha`, `beta`, dan riwayat state disimpan di VRAM dan
    /// dipakai langsung oleh [`Gpu::delta_bwd_dev`]; yang kembali ke host hanya `o`.
    /// Inilah bedanya dengan offload per operasi: kalau riwayat state harus diunduh
    /// dulu, biayanya ratusan MB per langkah dan lebih lambat daripada scan CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_scan(
        &self,
        q: &Mat,
        k: &Mat,
        v: &Mat,
        alpha: &Mat,
        beta: &Mat,
        batch: usize,
        seq: usize,
        n_heads: usize,
        dh: usize,
    ) -> Result<(Mat, DeltaDev), String> {
        let d = n_heads * dh;
        let t_len = batch * seq;
        assert_eq!(q.rows, t_len, "delta_scan: q rows harus batch*seq");
        let dq = DevBuf::new(q.d.len() * 4)?;
        let dk = DevBuf::new(k.d.len() * 4)?;
        let dv = DevBuf::new(v.d.len() * 4)?;
        let da = DevBuf::new(alpha.d.len() * 4)?;
        let db = DevBuf::new(beta.d.len() * 4)?;
        let dout = DevBuf::new(t_len * d * 4)?;
        let n_state = n_heads * batch * (seq + 1) * dh * dh;
        let dstates = DevBuf::new(n_state * 4)?;
        dq.upload(&q.d)?;
        dk.upload(&k.d)?;
        dv.upload(&v.d)?;
        da.upload(&alpha.d)?;
        db.upload(&beta.d)?;
        let (mut h_i, mut dh_i, mut b_i, mut seq_i) =
            (n_heads as i32, dh as i32, batch as i32, seq as i32);
        let shared = ((dh * dh + 3 * dh) * 4) as u32;
        self.launch(
            "delta_fwd_batch_f32",
            (n_heads as u32, batch as u32, 1),
            (128, 1, 1),
            shared,
            vec![
                &dq.0 as *const _ as *mut c_void,
                &dk.0 as *const _ as *mut c_void,
                &dv.0 as *const _ as *mut c_void,
                &da.0 as *const _ as *mut c_void,
                &db.0 as *const _ as *mut c_void,
                &dout.0 as *const _ as *mut c_void,
                &dstates.0 as *const _ as *mut c_void,
                &mut h_i as *mut _ as *mut c_void,
                &mut dh_i as *mut _ as *mut c_void,
                &mut b_i as *mut _ as *mut c_void,
                &mut seq_i as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        let mut out = vec![0.0f32; t_len * d];
        dout.download(&mut out)?;
        Ok((
            Mat::from_vec(t_len, d, out),
            DeltaDev {
                q: dq,
                k: dk,
                v: dv,
                alpha: da,
                beta: db,
                states: dstates,
                batch,
                seq,
                n_heads,
                dh,
            },
        ))
    }

    /// BPTT delta rule di device, memakai state dari [`Gpu::delta_scan`].
    ///
    /// Mengembalikan `(d_qkv, d_gate)`: gradien kolom q/k/v setelah backward
    /// L2-normalisasi, dan gradien logit gate setelah backward sigmoid. Yang naik ke
    /// host hanya dua matriks ini (T x 3d dan T x 2H), bukan riwayat state.
    pub fn delta_bwd_dev(
        &self,
        dev: &DeltaDev,
        d_o: &Mat,
        qnorm: &Mat,
        knorm: &Mat,
    ) -> Result<(Mat, Mat), String> {
        let (batch, seq, n_heads, dh) = (dev.batch, dev.seq, dev.n_heads, dev.dh);
        let d = n_heads * dh;
        let t_len = batch * seq;
        assert_eq!(d_o.rows, t_len);
        let ddo = DevBuf::new(d_o.d.len() * 4)?;
        let dqn = DevBuf::new(qnorm.d.len() * 4)?;
        let dkn = DevBuf::new(knorm.d.len() * 4)?;
        let d_dqkv = DevBuf::new(t_len * 3 * d * 4)?;
        let d_dgate = DevBuf::new(t_len * 2 * n_heads * 4)?;
        ddo.upload(&d_o.d)?;
        dqn.upload(&qnorm.d)?;
        dkn.upload(&knorm.d)?;
        let (mut h_i, mut dh_i, mut b_i, mut seq_i) =
            (n_heads as i32, dh as i32, batch as i32, seq as i32);
        let shared = ((2 * dh * dh + 9 * dh + 128) * 4) as u32;
        self.launch(
            "delta_bwd_f32",
            (n_heads as u32, batch as u32, 1),
            (128, 1, 1),
            shared,
            vec![
                &dev.states.0 as *const _ as *mut c_void,
                &dev.q.0 as *const _ as *mut c_void,
                &dev.k.0 as *const _ as *mut c_void,
                &dev.v.0 as *const _ as *mut c_void,
                &dev.alpha.0 as *const _ as *mut c_void,
                &dev.beta.0 as *const _ as *mut c_void,
                &dqn.0 as *const _ as *mut c_void,
                &dkn.0 as *const _ as *mut c_void,
                &ddo.0 as *const _ as *mut c_void,
                &d_dqkv.0 as *const _ as *mut c_void,
                &d_dgate.0 as *const _ as *mut c_void,
                &mut h_i as *mut _ as *mut c_void,
                &mut dh_i as *mut _ as *mut c_void,
                &mut b_i as *mut _ as *mut c_void,
                &mut seq_i as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        let mut dqkv = vec![0.0f32; t_len * 3 * d];
        d_dqkv.download(&mut dqkv)?;
        let mut dgate = vec![0.0f32; t_len * 2 * n_heads];
        d_dgate.download(&mut dgate)?;
        Ok((
            Mat::from_vec(t_len, 3 * d, dqkv),
            Mat::from_vec(t_len, 2 * n_heads, dgate),
        ))
    }
}

/// State delta rule yang tinggal di VRAM antara forward dan backward.
pub struct DeltaDev {
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    alpha: DevBuf,
    beta: DevBuf,
    states: DevBuf,
    pub batch: usize,
    pub seq: usize,
    pub n_heads: usize,
    pub dh: usize,
}

/// Logits kepala output yang tinggal di VRAM antara forward dan backward.
///
/// Yang disimpan bukan cuma `logits` (T x vocab) tapi juga `h_final` (T x d), karena
/// `d_head = h_final^T @ dlogits` membutuhkannya lagi dan mengunggahnya ulang hanya
/// membuang waktu.
pub struct LogitsDev {
    logits: DevBuf,
    h_final: DevBuf,
    t_len: usize,
    vocab: usize,
}

/// State MoE yang tinggal di VRAM antara forward dan backward.
///
/// `xn`, bobot `w1`/`w2`, dan `pre`/`hidden` per expert semuanya disimpan: forward sudah
/// menghitungnya, backward pasti membutuhkannya lagi. Dulu semuanya turun ke host lalu
/// naik kembali (30 MB per langkah) hanya untuk dipakai loop CPU.
pub struct MoeDev {
    xn: DevBuf,
    w1: DevBuf,
    w2: DevBuf,
    experts: Vec<MoeExpertDev>,
    t_len: usize,
    d: usize,
    d_ff: usize,
}

pub struct MoeExpertDev {
    idx: DevBuf,
    pre: DevBuf,
    hidden: DevBuf,
    n: usize,
}

/// Gradien MoE hasil `moe_bwd_dev`, semuanya sudah di host.
pub struct MoeBwd {
    pub d_w1: Mat,
    pub d_b1: Vec<f32>,
    pub d_w2: Mat,
    pub d_b2: Vec<f32>,
    pub dxn: Mat,
}

impl MoeDev {
    /// Unduh `pre`/`hidden` per expert. Hanya untuk jalur cadangan dan gate paritas —
    /// jalur GPU tidak membutuhkannya di host.
    pub fn download_pre_hidden(&self) -> Result<(Vec<Mat>, Vec<Mat>), String> {
        let mut pre = Vec::with_capacity(self.experts.len());
        let mut hidden = Vec::with_capacity(self.experts.len());
        for ex in self.experts.iter() {
            let mut p = vec![0.0f32; ex.n * self.d_ff];
            let mut h = vec![0.0f32; ex.n * self.d_ff];
            ex.pre.download(&mut p)?;
            ex.hidden.download(&mut h)?;
            pre.push(Mat::from_vec(ex.n, self.d_ff, p));
            hidden.push(Mat::from_vec(ex.n, self.d_ff, h));
        }
        Ok((pre, hidden))
    }
}

impl LogitsDev {
    /// Unduh logits ke host. Dipakai jalur evaluasi dan fallback kalau GPU gagal —
    /// bukan jalur training.
    pub fn to_host(&self) -> Result<Mat, String> {
        let mut out = vec![0.0f32; self.t_len * self.vocab];
        self.logits.download(&mut out)?;
        Ok(Mat::from_vec(self.t_len, self.vocab, out))
    }
}

impl DeltaDev {
    /// Unduh riwayat state. Hanya untuk uji paritas — jalur training memakai
    /// [`Gpu::delta_bwd_dev`] dan tidak pernah menyalinnya ke host.
    pub fn download_states(&self) -> Result<Vec<Vec<f32>>, String> {
        let n = self.n_heads * self.batch * (self.seq + 1) * self.dh * self.dh;
        let mut flat = vec![0.0f32; n];
        self.states.download(&mut flat)?;
        let per = (self.seq + 1) * self.dh * self.dh;
        Ok(flat.chunks(per).map(|c| c.to_vec()).collect())
    }
}

// CUDA context terikat ke thread pembuatnya. Engine ini satu thread, jadi `Gpu`
// ditandai Send+Sync secara eksplisit; kalau nanti dipakai multi-thread, context
// harus dibuat per thread.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

/// Alasan konkret kegagalan inisialisasi GPU, untuk diagnosis di mesin baru.
static INIT_ERR: OnceLock<String> = OnceLock::new();

/// `None` kalau tidak ada nvcc saat build, PTX gagal dimuat, atau tidak ada device.
pub fn gpu() -> Option<&'static Gpu> {
    GPU.get_or_init(Gpu::init).as_ref()
}

static ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static ENV_CHECKED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// `ASCA_GPU=0` mematikan jalur GPU tanpa membangun ulang — dipakai untuk mengukur
/// CPU dan GPU di binary yang sama.
///
/// Penting: `std::env::var` mengalokasi dan membaca environment setiap panggilan.
/// Versi pertama fungsi ini dipanggil dari dalam `Mat::matmul`, artinya satu lookup
/// environment per perkalian matriks — biaya itu sempat membuat matmul 5x lebih lambat
/// daripada benchmark terisolasi. Sekarang environment dibaca sekali saja.
pub fn available() -> bool {
    ENV_CHECKED.get_or_init(|| {
        if std::env::var("ASCA_GPU").ok().as_deref() == Some("0") {
            ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    ENABLED.load(std::sync::atomic::Ordering::Relaxed) && gpu().is_some()
}

pub fn set_enabled(v: bool) {
    ENABLED.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Apakah komponen bernama boleh di-offload.
///
/// `ASCA_GPU=0` mematikan seluruh jalur GPU. `ASCA_GPU_OFF=scan,moe` mematikan
/// komponen satu per satu, sehingga kontribusi tiap komponen bisa diukur A/B pada
/// binary yang sama — tanpa itu, membandingkan "semua GPU" lawan "semua CPU" hanya
/// memberi satu angka gabungan dan tidak bisa menyalahkan komponen mana pun.
/// Komponen yang **terukur lebih lambat** di GPU, jadi hanya dipakai kalau diminta
/// eksplisit lewat `ASCA_GPU_ON=adamw`.
///
/// AdamW masuk kategori ini: 3 salinan state optimizer (153 MB/langkah pada model 12,7M)
/// lebih mahal daripada 12,7 juta update di CPU — bahkan di Colab T4 yang hanya 2 vCPU
/// (terukur 1.866 vs 1.650 tok/s). Yang menang bukan memindahkan hitungannya ke GPU,
/// tapi berhenti menyalin state-nya; itu baru mungkin kalau seluruh model tinggal di
/// device.
pub fn offload_opt_in(component: &str) -> bool {
    static ON: OnceLock<Vec<String>> = OnceLock::new();
    let on = ON.get_or_init(|| {
        std::env::var("ASCA_GPU_ON")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    });
    available() && on.iter().any(|c| c == component)
}

pub fn offload(component: &str) -> bool {
    static OFF: OnceLock<Vec<String>> = OnceLock::new();
    let off = OFF.get_or_init(|| {
        std::env::var("ASCA_GPU_OFF")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    });
    available() && !off.iter().any(|c| c == component)
}

/// Deskripsi singkat untuk log.
pub fn describe() -> String {
    match gpu() {
        Some(g) => format!("GPU: {} ({} SM, {}, {})", g.name, g.sm_count, ARCH, NVCC),
        None => match INIT_ERR.get() {
            Some(e) => format!("GPU: tidak tersedia (arch target {ARCH}) - {e}"),
            None => format!("GPU: tidak tersedia (arch target {ARCH})"),
        },
    }
}

/// Ambang ukuran offload, **diukur** bukan ditebak.
///
/// `asca gpu` menunjukkan 128x64x192 = 1,6e6 hanya 0,90x (salinan lebih mahal), sedang
/// 512x64x1024 = 3,4e7 sudah 4x. Ambang 1e6 yang dipakai sebelumnya justru membuat
/// matmul kecil model (qkv 128x64x192) dilempar ke GPU dan 3x lebih lambat.
pub fn worth_offload(m: usize, k: usize, n: usize) -> bool {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let min = *MIN.get_or_init(|| {
        std::env::var("ASCA_GPU_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MIN_FLOPS)
    });
    m.saturating_mul(k).saturating_mul(n) >= min
}

/// Ambang bawaan. Dengan batching, matriks model berukuran 512 baris, dan di ukuran
/// itu GPU sudah menang (512x64x128 = 4,2e6 -> 1,68x; 512x64x192 = 6,3e6 -> 2,11x;
/// 512x64x1024 = 3,4e7 -> 3,95x). Pada 128 baris GPU baru menang di atas ~2e7, jadi
/// ambang ini bergantung pada batching dan harus diukur ulang kalau bentuk berubah.
pub const DEFAULT_MIN_FLOPS: usize = 4_000_000;
