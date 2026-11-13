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

## Yang belum dikerjakan

1. **Batch lintas sekuens.** Batch 4 masih diproses satu per satu, jadi setiap GEMM
   hanya 128 baris. Menggabungkannya menjadi 512 baris akan membuat GEMM 4× lebih besar
   dan membuka jalur GPU yang sesungguhnya.
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
