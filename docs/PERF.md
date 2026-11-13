# Optimasi performa CPU: 6.624 → 20.407 tok/s (3,08×)

Semua angka di sini diukur dengan `asca bench` pada engine yang sama, loss identik
sebelum dan sesudah setiap perubahan (2,0615 pada seed 0 / 100 langkah; 1,7343 pada
400 langkah / 3 seed). Artinya yang berubah hanya kecepatan, bukan matematika.

## Ringkasan

| model | sebelum | sesudah | rasio |
|---|---|---|---|
| ASCA v2 | 6.624 tok/s | **20.407 tok/s** | **3,08×** |
| dense (d_ff=128) | 15.899 tok/s | **16.370 tok/s** | 1,03× |

Dampaknya pada perbandingan: klaim lama "ASCA 2× lebih lambat pada active params sama"
**dicabut** — itu artefak kode yang belum dioptimasi, bukan sifat arsitektur. Sesudah
kedua jalur dioptimasi, ASCA **1,26× lebih cepat** pada active params sama dan
**4,53× lebih cepat** pada total params sama.

## Cara menemukannya

Tidak ada `perf` di mesin ini, jadi saya menambahkan profiler internal opt-in
(`ASCA_PROF=1`, `src/prof.rs`) yang mengakumulasi waktu per komponen. Keputusan
optimasi harus berangkat dari pengukuran; tebakan saya sebelumnya (bahwa matmul
mendominasi) salah — matmul hanya ~18% dari satu langkah.

Profil awal satu langkah (60 langkah × 4 sekuens, ms):

```
moe.fwd     534.1   57.7%
delta.qkv   272.5   29.4%
delta.scan   72.1    7.8%
head         28.2    3.0%
delta.out    18.4    2.0%
```

## Tujuh penyebab, berurutan menurut dampak

### 1. `std::env::var` di dalam `Mat::matmul`

Jalur GPU dicek setiap perkalian matriks, dan `available()` membaca environment
(`ASCA_GPU`) setiap kali. `env::var` mengalokasi `String` dan menelusuri environment —
satu lookup per matmul. Diperbaiki dengan `OnceLock`: environment dibaca sekali.

### 2. Ambang offload GPU terlalu rendah

Ambang `M*K*N >= 1e6` membuat matmul kecil model (qkv 128×64×192 = 1,6e6) dilempar ke
GPU, padahal `asca gpu` menunjukkan ukuran itu hanya **0,90×** — salinan H2D/D2H lebih
mahal daripada matmul-nya. Ambang dinaikkan ke `2e7` berdasarkan pengukuran (512×64×1024
= 3,4e7 sudah 4×).

### 3. `HashMoe::forward` memakai akses ber-bound-check di loop terdalam

```rust
// sebelum: satu perkalian + dua batas array per elemen
for j in 0..d_ff {
    let col = e * d_ff + j;
    let mut acc = self.b1.value.at(0, col);
    for c in 0..d { acc += row[c] * self.w1.value.at(c, col); }
```

Diganti dengan slice baris bersambung **dan** pengelompokan per expert: token
dikelompokkan menurut `hash(id) % n_experts`, lalu loop bobot ditaruh di luar sehingga
satu baris `w1`/`w2` dibaca sekali dan dipakai untuk semua token expert itu.

### 4. `HashMoe::backward` menghitung ulang pra-aktivasi + akumulasi gradien lewat `set`/`at`

Versi lama merekonstruksi pra-aktivasi SiLU dengan `self.w1.value.at(c, col)` di dalam
loop per token — satu pass penuh GEMM yang dibuang — dan mengakumulasi gradien dengan
`grad.set(r, c, grad.at(r, c) + ...)`. Sekarang `pre` disimpan di cache, dan semua
akumulasi memakai slice baris. Ini penyumbang terbesar: `moe.bwd` 631 → 511 ms, dan
total langkah turun 2,8×.

### 5. Scan delta mengalokasi `Vec` per timestep

`sk` dan `u` dibuat ulang di setiap langkah waktu: 2 × T × H per layer-forward. Sekarang
dialokasikan sekali per head. `delta.scan` 72 → 60 ms.

### 6. Transpose yang tidak perlu di backward

`x.transpose().matmul(y)` mengalokasi dan menyalin matriks transpose lebih dulu. Ada dua
pengganti: `Mat::matmul_ta` (baru, `selfᵀ @ other` tanpa alokasi) dan `Mat::matmul_t`
yang sudah ada (`self @ otherᵀ`). Dipakai di `dW_o`, `d_head`, `d_hidden`, dan embedding.

### 7. Akumulasi gradien embedding lewat `set`/`at`

Diganti slice baris.

Baseline dense mendapat perlakuan yang sama (menyimpan `pre`, `matmul_t` untuk
`d_hidden`, slice baris di backward attention), supaya perbandingan tidak berat sebelah:
dense 15.899 → 16.370 tok/s.

## Profil sesudah optimasi

```
delta.bwd   416.2   27.0%
moe.bwd     511.2   33.1%
other       200.0   13.0%   (backward kepala output)
delta.qkv   168.5   10.9%
moe.fwd     123.9    8.0%
delta.scan   59.7    3.9%
head         38.5    2.5%
delta.out    17.9    1.2%
norm          6.3    0.4%
```

Backward sekarang mendominasi (~73%), sesuai harapan untuk model kecil: biaya forward
dan backward sebanding, tetapi backward punya lebih banyak akumulasi gradien.

## Batching lintas sekuens (dan apa yang benar-benar terjadi)

Batch 4 sekarang diproses sebagai satu matriks `B*T` baris: rekurensi delta dan attention
tetap per sekuens (block-diagonal), tetapi semua GEMM memakai 512 baris.

Gate paritas: `tests/batch_parity.rs` membandingkan satu forward/backward `batch=4`
dengan empat `batch=1` yang dirata-ratakan. Loss dan gradien cocok sampai **4,3e-7**
(ASCA) dan **5,2e-7** (dense) — artinya tidak ada token yang bocor antar sekuens.

Hasilnya **tidak seperti yang saya harapkan**:

| model | sebelum batching | sesudah batching | perubahan |
|---|---|---|---|
| ASCA (total 331k) | 20.407 tok/s | 20.410 tok/s | ~0% |
| ASCA (active 99k) | 20.407 tok/s | 20.320 tok/s | ~0% |
| dense d_ff=1024 | 4.503 tok/s | **6.130 tok/s** | **+36%** |
| dense d_ff=128 | 16.370 tok/s | **21.493 tok/s** | **+31%** |

Batching membantu **dense**, bukan ASCA. Alasannya: dense menghabiskan waktunya di GEMM
besar (attention + FFN) yang langsung menikmati 512 baris dan ambang GPU, sedangkan biaya
ASCA didominasi kerja per-token (bucket MoE, scan delta) yang tidak menjadi lebih besar
ketika baris ditambah. Untuk ASCA batching bahkan sedikit merugikan (−0,4%), mungkin
karena working set yang lebih besar.

### Ambang offload GPU harus dikalibrasi ulang

Ambang `2e7` sebelumnya diukur untuk matriks 128 baris. Pada 512 baris GPU sudah menang
jauh lebih awal:

| bentuk | CPU ms | GPU ms | speedup |
|---|---|---|---|
| 512×64×128 | 0,272 | 0,162 | 1,68× |
| 512×64×192 | 0,422 | 0,200 | 2,11× |
| 512×64×1024 | 2,583 | 0,655 | 3,95× |

Ambang bawaan kini **4e6**, bisa diatur lewat `ASCA_GPU_MIN`. Dengan ambang itu
dense-active naik 16.673 → 21.493 tok/s.

### Dampak pada perbandingan

| pencocokan | ASCA | dense | rasio |
|---|---|---|---|
| total params 331k | 20.410 tok/s | 6.130 tok/s | **3,33×** (sebelumnya 4,53×) |
| active params 99k | 20.320 tok/s | 21.493 tok/s | **0,95×** (sebelumnya 1,26×) |

Klaim "ASCA 1,26× lebih cepat pada active params sama" **digantikan**: setelah kedua
arsitektur mendapat batching dan offload GPU, throughput-nya **setara** (selisih 5%).
Keunggulan ASCA yang tersisa pada mode ini adalah **state decode** (2.048 float konstan
vs 32.768 yang tumbuh O(T)), bukan kecepatan training.

## Kernel SIMD AVX2/FMA (`src/simd.rs`)

Loop terdalam sebelumnya skalar. Sekarang ada kernel AVX2+FMA untuk `axpy`, `dot`,
`scale`, `scale_axpy`, dan `axpy2`, dengan deteksi fitur saat runtime (`is_x86_feature_detected!`,
std saja) sehingga binary yang sama tetap jalan di mesin tanpa AVX2 lewat jalur skalar.
Dua unit test membandingkan jalur AVX2 dengan jalur skalar untuk panjang 0, 1, 7, 8, 9, 33,
dan 128.

Dipasang di: `Mat::matmul`, `Mat::matmul_ta`, `Mat::matmul_t`, `rmsnorm_rows` (+backward),
loop MoE forward/backward, scan delta forward/backward, dan backward attention dense.

Hasil (400 langkah, 3 seed, lr 6e-3, `ASCA_GPU_MIN=4e6`):

| model | sebelum SIMD | sesudah SIMD | perubahan |
|---|---|---|---|
| ASCA (total) | 20.410 tok/s | **32.174 tok/s** | **+58%** |
| ASCA (active) | 20.320 tok/s | **33.844 tok/s** | **+67%** |
| dense d_ff=1024 | 6.130 tok/s | 5.933 tok/s | ~0% |
| dense d_ff=128 | 21.493 tok/s | 21.219 tok/s | ~0% |

Loss **tidak berubah sama sekali** (1,7343 / 1,9762 / 1,9778), jadi ini murni kecepatan.

### A/B tanpa membangun ulang

`ASCA_NO_SIMD=1` memaksa jalur skalar, jadi kontribusi SIMD bisa diukur pada binary yang
sama dengan run bergantian (penting kalau mesin sedang dipakai proses lain):

| run | SIMD | skalar | rasio |
|---|---|---|---|
| 1 | 27.285 tok/s | 19.894 tok/s | 1,37× |
| 2 | 28.195 tok/s | 19.067 tok/s | 1,48× |
| 3 | 26.799 tok/s | 18.130 tok/s | 1,48× |

> **Koreksi.** Sempat saya duga SIMD merugikan loop pendek (16 float) di scan delta,
> karena `delta.scan` di profil naik 60 → 112 ms. Itu artefak: profil lama memakai
> penyebut 240 forward 128 baris, yang baru 40 forward 512 baris. Setelah dinormalisasi
> per token, dan dikonfirmasi A/B di atas, SIMD menguntungkan di semua loop.

### Kenapa dense tidak ikut naik

Karena matmul dense sudah dilempar ke GPU (512×64×128 = 4,2e6 sudah di atas ambang 4e6),
sehingga perbaikan matmul CPU tidak menyentuhnya. ASCA tidak bisa begitu: biayanya ada di
kerja **per-token** (bucket MoE, scan delta) yang tidak punya bentuk GEMM besar untuk
di-offload, jadi vektorisasi CPU langsung terasa. Ini juga menjelaskan kenapa offload GPU
sebagian **menutupi** perbaikan CPU pada baseline — kalau hanya salah satu jalur yang
dioptimasi, rasionya menyesatkan.

| pencocokan | ASCA | dense | rasio |
|---|---|---|---|
| total params 331k | 29.696 tok/s | 13.131 tok/s | **2,26×** |
| active params 99k | 29.195 tok/s | 26.430 tok/s | **1,10×** |

> Angka 5,42× / 1,59× yang sempat ditulis di sini **dicabut**. Keduanya diukur saat
> baseline dense masih memakai kode yang belum dioptimasi (lihat bagian berikut).

## Baseline dense juga harus dioptimasi

Rasio throughput hanya bermakna kalau kedua sisi memakai engine yang sama-sama dirawat.
Setelah ASCA dioptimasi 3×, baseline dense tertinggal di 5.933 tok/s — dan itu bukan
sifat arsitekturnya, melainkan dua bug performa di `dense.rs`:

1. **Akumulasi nilai attention** memakai `v.at(base + s, off + c)` di loop terdalam:
   `dh x t` akses ber-bounds-check untuk setiap token. Diganti pembacaan slice + `axpy`.
2. **`d_pre` merekonstruksi pra-aktivasi** dengan loop `.at()` (`t_len x d_ff x d` akses)
   padahal nilainya **sudah tersimpan di `DenseLayerCache.pre`** sejak forward. Diganti
   pembacaan slice. Ini pola yang sama persis dengan bug MoE sebelumnya: nilai ada di
   cache, tapi dihitung ulang.

Hasil: dense **4.825 → 13.131 tok/s (2,7×)**, loss identik sampai digit terakhir
(2,2135 pada 200 langkah, 1,9764 pada 400 langkah/3 seed).

### Jebakan `matmul_ta`

Sempat saya ganti enam situs `a.transpose().matmul(b)` menjadi `a.matmul_ta(b)` di
backward dense. Hasilnya **lebih lambat**: 12.313 → 9.430 tok/s. Sebabnya `matmul_ta`
tidak punya jalur GPU, jadi matmul besar yang tadinya di-offload jatuh ke CPU. Guard
hybrid sudah ditambahkan ke `matmul_ta` (bentuk besar didelegasikan ke `matmul`), tetapi
untuk pola-pola ini transpose + `matmul` tetap lebih cepat (pola akses lebih ramah cache),
jadi penggantiannya dibatalkan. `matmul_t` bahkan tidak punya jalur GPU sama sekali.

## Decode: ASCA datar, dense tumbuh O(T)

`asca decode` mendekode satu token demi satu dan mengukur ms/token pada beberapa panjang
konteks. Kedua arsitektur memakai engine yang sama, f32, CPU, matmul dan RMSNorm yang sama.

```bash
./target/release/asca decode --ctx 128,512,1024,2048
./target/release/asca decode --model dense --ctx 128,512,1024,2048
```

| konteks | ASCA ms/token | dense ms/token | state ASCA | state dense |
|---|---|---|---|---|
| 128 | **0,008** | 0,022 | 2.048 | 43.008 |
| 512 | **0,008** | 0,075 | 2.048 | 141.312 |
| 1.024 | **0,009** | 0,141 | 2.048 | 272.384 |
| 2.048 | **0,010** | 0,295 | 2.048 | 534.528 |

Pada konteks 2.048: **29× lebih cepat per token** dan **261× lebih kecil state-nya**.

Mekanismenya bukan tuning. ASCA memperbarui `S` berukuran `dh x dh` per head di tempat,
jadi biaya per token konstan. Dense harus membaca seluruh K dan V setiap token, jadi
biayanya linear terhadap konteks — dan KV cache-nya tumbuh linear juga.

Kapan keunggulan ini bertahan: state ASCA = `n_layers * n_heads * (d/h)^2 = n_layers * d^2/h`,
KV dense = `n_layers * T * d * 2`. ASCA menang begitu `d/h < 2T`, yang untuk semua
konteks realistis selalu benar. Yang berubah saat model diperbesar hanyalah besar
keunggulannya, bukan arahnya.

Batas klaim: model masih tiny (`d=64`, 2 layer, 4 head) dan ini CPU f32. Angka absolutnya
tidak akan berlaku di model besar; yang berlaku adalah bentuk kurvanya (datar vs linear)
dan rasio ukuran state.

## Yang belum dikerjakan

1. ~~Batch lintas sekuens~~ **selesai** — lihat bagian di atas; hasilnya netral untuk
   ASCA dan +31–36% untuk dense.
2. **Fusi elementwise.** RMSNorm, residual, sigmoid, dan gate masih beberapa pass
   terpisah dengan alokasi `Mat` di antaranya.
3. **Kernel GPU untuk seluruh blok**, bukan per operasi — lihat [`CUDA.md`](CUDA.md).
4. **Matmul CPU yang di-block + SIMD.** Sekarang ~24 GFLOPS; pustaka seperti
   `matrixmultiply` bisa 3–5× lebih tinggi, tetapi menambah dependensi.

## Cara mengukur ulang

```bash
ASCA_PROF=1 rust/target/release/asca bench --data "$(./scripts/fetch_corpus.sh)" \
    --steps 60 --seeds 0 --match active --eval-batches 1
```
