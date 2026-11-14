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

use crate::tensor::Mat;

include!(concat!(env!("OUT_DIR"), "/gpu_build.rs"));

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

    pub fn delta_fwd(
        &self,
        q: &Mat,
        k: &Mat,
        v: &Mat,
        alpha: &[f32],
        beta: &[f32],
        t_len: usize,
        n_heads: usize,
        dh: usize,
    ) -> Result<Mat, String> {
        let d = n_heads * dh;
        let dq = DevBuf::new(q.d.len() * 4)?;
        let dk = DevBuf::new(k.d.len() * 4)?;
        let dv = DevBuf::new(v.d.len() * 4)?;
        let da = DevBuf::new(alpha.len() * 4)?;
        let db = DevBuf::new(beta.len() * 4)?;
        let dout = DevBuf::new(t_len * d * 4)?;
        dq.upload(&q.d)?;
        dk.upload(&k.d)?;
        dv.upload(&v.d)?;
        da.upload(alpha)?;
        db.upload(beta)?;
        let (mut t, mut h, mut dhv) = (t_len as i32, n_heads as i32, dh as i32);
        let shared = ((dh * dh + 3 * dh) * 4) as u32;
        self.launch(
            "delta_fwd_f32",
            (n_heads as u32, 1, 1),
            (128, 1, 1),
            shared,
            vec![
                &dq.0 as *const _ as *mut c_void,
                &dk.0 as *const _ as *mut c_void,
                &dv.0 as *const _ as *mut c_void,
                &da.0 as *const _ as *mut c_void,
                &db.0 as *const _ as *mut c_void,
                &dout.0 as *const _ as *mut c_void,
                &mut t as *mut _ as *mut c_void,
                &mut h as *mut _ as *mut c_void,
                &mut dhv as *mut _ as *mut c_void,
            ],
        )?;
        unsafe { sys::cuCtxSynchronize() };
        let mut out = vec![0.0f32; t_len * d];
        dout.download(&mut out)?;
        Ok(Mat::from_vec(t_len, d, out))
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
