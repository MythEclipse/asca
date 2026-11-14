# Kernel CUDA v2

Status: **blok delta sudah mempercepat training (3.530 → 3.967 tok/s); sisanya belum.**
Angka di bawah menjelaskan apa yang sudah jalan, apa yang gagal, dan apa yang harus
dikerjakan berikutnya.

## 1. Yang dibangun

Tanpa crate eksternal, mengikuti pola repo `llm` Anda: `nvcc -ptx` di `build.rs`, PTX
di-embed ke binary, driver API diikat langsung lewat `#[link(name = "cuda")]`.

| Berkas | Isi |
|---|---|
| `build.rs` | kompilasi `kernels/asca.cu` → PTX (`-O3`, `-arch=sm_89`), tulis `gpu_build.rs` berisi PTX + daftar kernel + versi nvcc. Tanpa `nvcc` crate tetap terbangun, jalur GPU dilaporkan tidak tersedia |
| `src/cuda/sys.rs` | binding driver API: context, modul PTX, buffer device, `cuLaunchKernel`, `cuFuncSetAttribute` |
| `src/cuda/mod.rs` | `Gpu` (lazy `OnceLock`), `matmul`, `matmul_ta`, `rmsnorm`, `delta_scan` + `delta_bwd_dev`, `DevBuf` (alokator caching), penghitung lalu lintas H2D/D2H; `ASCA_GPU=0` mematikan jalur GPU, `ASCA_GPU_OFF=scan,moe` mematikan per komponen tanpa rebuild |
| `kernels/asca.cu` | 15 kernel: `matmul_f32`, `matmul_ta_f32`, `add_bias_f32`, `axpy_f32`, `scale_f32`, `rmsnorm_f32`, `rmsnorm_bwd_f32`, `delta_fwd_batch_f32`, `delta_bwd_f32`, `cross_entropy_bwd_f32`, `adamw_f32`, `gather_rows_f32`, `scatter_rows_f32`, `silu_f32` |

`asca gpu` mencetak paritas dan benchmark. Kernel delta rule dipetakan **satu blok per
(head, sekuens)**, state `dh×dh` di shared memory, seluruh loop waktu di dalam kernel —
tanpa keluaran antara ke memori global.

## 2. Paritas (RTX 4050 Laptop, sm_89, CUDA 13.4)

| uji | galat relatif |
|---|---|
| `matmul` (64×64@64×192, 128×64@64×512, 37×53@53×91, 256³) | ≤ 2,6e-7 |
| `matmul_ta` (A^T @ B) | ≤ 2,6e-7 |
| `delta_fwd_batch_f32` (B=3, seq=24, H=4, dh=16) vs referensi CPU | 1,5e-7 (o), 1,1e-7 (state) |
| isolasi sekuens: `batch=3` vs 3× `batch=1` | **0,0** (bit-identik) |
| `delta_bwd_f32` vs BPTT Rust (B=8, seq=128, d=192, H=6) | ≤ 8,0e-7 relatif di semua parameter + `dx` |

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

## 3b. Blok delta pindah ke GPU: 3.530 → 3.967 tok/s

Ini hasil pertama di repo ini yang membuat GPU benar-benar mempercepat training — dan
jalan ke sana dimulai dari satu percobaan yang terukur **lebih lambat**.

### Percobaan pertama: scan saja — LEBIH LAMBAT

Rencana awalnya hanya memindahkan `delta.scan` (15% langkah di profil Colab), karena
kernelnya sudah ada dan sudah lolos paritas. A/B pada binary yang sama
(`ASCA_GPU_OFF=scan`), 12,7M params, batch 8 × seq 256, RTX 4050:

| | scan CPU | scan GPU |
|---|---|---|
| tok/s | **3.516** | 3.265 |
| `delta.scan` | 44,5 ms/langkah | 92,5 ms/langkah |
| D2H per langkah | 589 MB | **902 MB** |

Sebabnya bukan kernelnya: BPTT masih di CPU, jadi riwayat state `(seq+1)·dh²` per
(head, sekuens) harus diunduh — **+313 MB per langkah**. Itu kesalahan yang sama persis
dengan `scatter_rows` di MoE: memindahkan hitungan ke GPU tapi tetap pulang-pergi data
hanya memindahkan biayanya. Offload scan per operasi **dicabut**, bukan dioptimasi.

### Perbaikan: state tinggal di VRAM, BPTT ikut pindah

`delta_fwd_batch_f32` menulis riwayat state ke buffer device yang hidup di `DeltaDev`,
dan `delta_bwd_f32` membacanya langsung — tidak ada satu byte state pun yang menyeberang
ke host. Yang naik ke host hanya `o`, `d_qkv`, dan `d_gate`.

| | sebelum | sesudah |
|---|---|---|
| tok/s | 3.530 | **3.967** (+12,4%) |
| `delta.scan` | 40,6 ms/langkah | **12,4** |
| `delta.bwd` | 106,9 ms/langkah | **93,6** |
| D2H per langkah | 589 MB | 627 MB |

Loss identik sampai digit terakhir (6,1714 pada 20 langkah). Gate paritas: `o` 1,5e-7,
state 1,1e-7, isolasi sekuens bit-identik, gradien ≤ 8,0e-7 relatif.

Catatan jujur: `delta.bwd` hanya turun 13 ms karena BPTT CPU bukan bagian terbesar di
dalamnya — sisanya matmul dan akumulasi bias di host. Kernel BPTT-nya sendiri
~12 ms/langkah (48 blok, 256 langkah sekuensial, ~10 `__syncthreads` per langkah) ditambah
~10 ms transfer; BPTT CPU yang digantikannya ~35 ms.

### Di T4 (target sebenarnya)

Angka di atas dari laptop (20 core, RTX 4050). Di Colab T4 dengan 2 vCPU, scan dan BPTT
CPU jadi jauh lebih mahal, sehingga keuntungannya jauh lebih besar. Empat konfigurasi,
config identik (12,7M params, batch 8 x seq 256, 20 langkah, corpus tokenized 12.288):

| konfigurasi | tok/s |
|---|---|
| MoE CPU + delta CPU | 907 |
| delta CPU, MoE GPU | 1.010 |
| MoE CPU, delta GPU | 1.158 |
| **MoE GPU + delta GPU** | **1.374** |

Jadi blok delta bernilai **+36%** di T4 (1.010 → 1.374) versus +12,4% di laptop — persis
karena 2 vCPU. Catatan: 1.010 tok/s cocok dengan profil Colab Anda sebelumnya (1.055),
jadi pembandingnya nyata.

Dua hal yang perlu dicatat jujur:

- **Angka 1.628 tok/s "sebelumnya" belum terjelaskan.** Keempat konfigurasi di atas
  memakai config yang sama dan tidak ada yang mencapainya; kemungkinan besar config atau
  corpus pengukurannya berbeda (seq_len/batch/eval). Sebelum menyebut ini berhasil,
  konfigurasi 1.628 itu perlu direproduksi — kalau tidak, pembandingnya apel vs jeruk.
- **MoE forward di GPU tetap menang di T4** meski `scatter_rows` membuang salinan
  (1.158 → 1.374, +18,6%). Keluhan Anda soal 144 MB/langkah itu benar sebagai biaya, tapi
  di 2 vCPU biaya itu masih lebih murah daripada MoE di CPU. Di laptop (20 core)
  urutannya bisa terbalik — karena itu `ASCA_GPU_OFF=moe` sekarang ada, supaya bisa diukur
  per mesin, bukan ditebak.

### 3c. Kepala output tinggal di device — dan AdamW yang justru kalah

Dua kandidat berikutnya dari data lalu lintas: kepala output (~500 MB/langkah) dan AdamW
(~357 MB/langkah). Yang pertama menang, yang kedua **kalah** — dan angka negatifnya
dicatat di sini karena itu yang mencegah jalur lambat dipertahankan.

**Kepala output.** `logits` (T x vocab, 100 MB) dulu turun setelah GEMM kepala, naik lagi
untuk cross-entropy, turun sebagai `dlogits`, lalu naik dua kali untuk dua GEMM kepala di
backward. Sekarang `head_forward_dev` menyimpannya di VRAM dan `head_bwd_dev` menjalankan
cross-entropy + `d_head = h_final^T dlogits` + `dh = dlogits head^T` di sana. Hasilnya
**bit-identik** (loss, `d_head`, `dh` selisih 0,0e0) dan lalu lintasnya 523 MB → 31 MB per
langkah (T=2048).

| | laptop (20 core) | T4 (2 vCPU) |
|---|---|---|
| kepala di host | 4.088 tok/s | 1.443 tok/s |
| kepala di device | **4.837** (+18,3%) | **1.650** (+14,4%) |

**AdamW.** Momen `m`/`v` (2 x 50,9 MB) bisa tinggal di VRAM, jadi salinannya turun dari 7
buffer per parameter menjadi 3. Tapi tetap **lebih lambat daripada CPU**, bahkan di T4:

| AdamW | laptop | T4 |
|---|---|---|
| per operasi (7 salinan, kode lama) | 4.837 → turun | — |
| momen di VRAM (3 salinan) | 5.145 tok/s | 1.650 tok/s |
| CPU (12,7 juta update) | **5.198 tok/s** | **1.866 tok/s** |

Jadi jalur AdamW GPU yang lama itu **regresi**: 153 MB/langkah lebih mahal daripada
12,7 juta update di CPU, bahkan dengan 2 vCPU. Sekarang AdamW jalan di CPU secara default
dan jalur GPU-nya hanya aktif kalau diminta (`ASCA_GPU_ON=adamw`). Pelajarannya sama
dengan `scatter_rows`: memindahkan hitungan ke GPU tidak menolong kalau state-nya tetap
harus menyeberang. Yang menang adalah berhenti menyalin — dan itu baru mungkin kalau
seluruh model tinggal di device.

### Total sejauh ini

| | awal | sekarang | perubahan |
|---|---|---|---|
| laptop (12,7M, b8 x s256) | 3.530 tok/s | **5.198** | +47% |
| T4 (config sama) | 1.010 tok/s | **1.866** | +85% |
| lalu lintas per langkah | 1.602 MB | **743 MB** | -54% |

Loss di semua konfigurasi tetap 6,1714 (20 langkah) dan 6,1421–6,1422 (val). Sisa 743 MB
per langkah berasal dari parameter yang diunggah ulang oleh setiap operasi dan dari
aktivasi antar layer — keduanya baru bisa hilang dengan memindahkan seluruh model ke
device, bukan per komponen.

### Bug yang ditemukan gate-nya sendiri

Kernel `delta_bwd_f32` versi pertama menghasilkan galat **0,87 relatif** — bukan beda
pembulatan, tapi salah. Sebabnya: `q` dan `k` yang dikirim ke kernel **sudah**
dinormalisasi L2 di `forward`, dan kernel mengalikannya dengan `1/norma` sekali lagi.
Tanpa gate gradien GPU-vs-CPU di `asca gpu`, ini akan lolos: loss tetap turun, hanya lebih
buruk, dan tidak ada yang gagal.

### Ke mana sisa 1,6 GB per langkah

Penghitung di `DevBuf` membuat lalu lintas device bisa dibaca langsung (`ASCA_PROF=1`):

| arah | per langkah | penyumbang terbesar |
|---|---|---|
| H2D | 975 MB | AdamW (4 × 50,9 MB state), logits/dlogits (100 MB per salinan) |
| D2H | 627 MB | AdamW (3 × 50,9 MB), logits + dlogits (200 MB) |
| kernel | 350 | — |

Sisa pekerjaan terbesar karena itu bukan lagi MoE backward (96 ms/langkah CPU),
melainkan **kepala output + AdamW yang masih pulang-pergi tiap langkah**: satu `logits`
100 MB disalin ~4×, dan state optimizer 12,7M parameter disalin 7×. Itu ~900 MB/langkah
yang bisa hilang tanpa satu kernel baru — cukup dengan menahan tensor di device.

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

1. ~~**Tensor tinggal di device** untuk blok delta~~ **selesai** (§3b) — dan itu memang
   yang membuatnya menang. Sisa: kepala output, cross-entropy, dan AdamW masih
   pulang-pergi (975 MB + 627 MB per langkah).
2. **Fusi elementwise.** RMSNorm + residual + sigmoid + gate dalam satu kernel per blok,
   bukan 6 peluncuran terpisah.
3. **Batch lintas sekuens.** Gabungkan batch 4 menjadi 512 baris, sehingga GEMM jadi
   4× lebih besar — di situlah GPU mulai menang (4,14× pada 512 baris).
4. ~~**Backward delta di GPU** (`delta_bwd_f32`)~~ **selesai** (§3b), termasuk BPTT,
   backward L2-normalisasi, dan backward sigmoid dalam satu kernel.
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
