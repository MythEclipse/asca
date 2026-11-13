# Kernel CUDA v2

Status: **infrastruktur dan kernel benar, tetapi belum mempercepat training.** Angka di
bawah menjelaskan kenapa, dan apa yang harus dikerjakan supaya benar-benar cepat.

## 1. Yang dibangun

Tanpa crate eksternal, mengikuti pola repo `llm` Anda: `nvcc -ptx` di `build.rs`, PTX
di-embed ke binary, driver API diikat langsung lewat `#[link(name = "cuda")]`.

| Berkas | Isi |
|---|---|
| `build.rs` | kompilasi `kernels/asca.cu` → PTX (`-O3`, `-arch=sm_89`), tulis `gpu_build.rs` berisi PTX + daftar kernel + versi nvcc. Tanpa `nvcc` crate tetap terbangun, jalur GPU dilaporkan tidak tersedia |
| `src/cuda/sys.rs` | binding driver API: context, modul PTX, buffer device, `cuLaunchKernel`, `cuFuncSetAttribute` |
| `src/cuda/mod.rs` | `Gpu` (lazy `OnceLock`), `matmul`, `matmul_ta`, `rmsnorm`, `delta_fwd`; `ASCA_GPU=0` mematikan jalur GPU tanpa rebuild |
| `kernels/asca.cu` | 8 kernel: `matmul_f32`, `matmul_ta_f32`, `add_bias_f32`, `axpy_f32`, `scale_f32`, `rmsnorm_f32`, `rmsnorm_bwd_f32`, `delta_fwd_f32` |

`asca gpu` mencetak paritas dan benchmark. Kernel delta rule dipetakan **satu blok per
head**, state `dh×dh` di shared memory, seluruh loop waktu di dalam kernel — tanpa
keluaran antara ke memori global.

## 2. Paritas (RTX 4050 Laptop, sm_89, CUDA 13.4)

| uji | galat relatif |
|---|---|
| `matmul` (64×64@64×192, 128×64@64×512, 37×53@53×91, 256³) | ≤ 2,6e-7 |
| `matmul_ta` (A^T @ B) | ≤ 2,6e-7 |
| `delta_fwd` (T=24, H=4, dh=16) vs referensi CPU | 1,4e-7 |

End-to-end: loss training **identik** dengan jalur CPU (`2,0615` pada seed 0, 100 langkah),
jadi offload matmul tidak mengubah matematika.

## 3. Kecepatan GEMM

| M | K | N | CPU ms | GPU ms | CPU GFLOPS | GPU GFLOPS | speedup |
|---|---|---|---|---|---|---|---|
| 128 | 64 | 192 | 0,112 | 0,115 | 28,1 | 27,3 | 0,97× |
| 512 | 64 | 1024 | 2,741 | 0,662 | 24,5 | 101,4 | **4,14×** |
| 1024 | 1024 | 1024 | 90,746 | 5,208 | 23,7 | 412,3 | **17,4×** |
| 4096 | 1024 | 1024 | 368,548 | 17,954 | 23,3 | 478,4 | **20,5×** |

Kernel GEMM-nya masih naif (tile 32×32, f32, tanpa tensor core), jadi 478 GFLOPS itu
sekitar 4% puncak f32 kartu ini — masih ada ruang besar, tapi bukan itu hambatannya.

## 4. Kenapa training belum cepat

Diukur, bukan diduga:

- Training ASCA saat itu: **6.856 tok/s** CPU vs **6.829 tok/s** GPU (100 langkah, batch 4,
  seq 128). Praktis nol. Dense: 14.557 → 15.549 tok/s (+7%).
- Satu langkah saat itu = 512 token / 6.856 tok/s ≈ **74,7 ms**.
- FLOPs per langkah ≈ 6 · N_aktif · T = 6 · 99.344 · 512 ≈ **305 MFLOP**.
- Pada 23,7 GFLOPS (matmul CPU terukur), itu ≈ **13 ms** — hanya **~18%** dari waktu
  langkah pada saat itu.

> **Premis ini berubah setelah [`PERF.md`](PERF.md).** Jalur CPU naik 3,08× menjadi
> **20.407 tok/s**, jadi satu langkah kini ≈ **25 ms** dan porsi matmul di dalamnya
> relatif lebih besar, tetapi offload per-operasi tetap tidak menarik karena setiap
> panggilan masih membayar satu salinan H2D + D2H + sinkronisasi. Kesimpulan yang sama
> berlaku lebih kuat: yang dibutuhkan adalah tensor yang tinggal di device dan kernel
> yang menyatukan satu blok penuh, bukan GEMM yang lebih cepat untuk matriks 128×64.

Jadi bahkan kalau matmul menjadi instan, langkah hanya turun ~18%. Sisanya: RMSNorm,
sigmoid, penjumlahan residual, scan delta, dan **alokasi `Mat` per operasi** di kode Rust.
Selain itu setiap panggilan GPU sekarang menyalin H2D + D2H + `cuCtxSynchronize`; pada
matriks 128×64@64×192 salinan itu lebih mahal daripada matmul-nya (0,97×).

## 5. Yang harus dikerjakan supaya angka ini berubah

Berurutan, dari yang paling menentukan:

1. **Tensor tinggal di device.** Satu upload per langkah, satu download; semua
   intermediate tetap di GPU. Ini menghapus ~90% salinan dan sinkronisasi.
2. **Fusi elementwise.** RMSNorm + residual + sigmoid + gate dalam satu kernel per blok,
   bukan 6 peluncuran terpisah.
3. **Batch lintas sekuens.** Gabungkan batch 4 menjadi 512 baris, sehingga GEMM jadi
   4× lebih besar — di situlah GPU mulai menang (4,14× pada 512 baris).
4. **Backward delta di GPU** (`delta_bwd_f32`), termasuk BPTT yang sekarang ditulis tangan
   di Rust.
5. Setelah itu baru tensor core f16 dan tile 64×64.

Perkiraan realistis setelah langkah 1–4: 3–6× lebih cepat, artinya run 4M token × 5 seed
turun dari ~2 jam menjadi ~20–40 menit.

## 6. Cara pakai sekarang

```bash
cd rust
cargo build --release
./target/release/asca gpu                    # paritas + benchmark GEMM
ASCA_GPU=0 ./target/release/asca train ...   # paksa jalur CPU
ASCA_CUDA_ARCH=sm_90 cargo build --release   # arsitektur lain
```

Ambang offload saat ini `M*K*N >= 1e6` (di bawah itu salinan lebih mahal). Ambang ini
sengaja konservatif supaya jalur GPU tidak pernah memperlambat training.
