# Hasil v2: ASCA vs baseline dense di engine yang sama

Status: **hasil pertama yang metodologisnya sah**, tapi skalanya masih kecil. Belum
boleh diklaim sebagai keunggulan arsitektur (lihat §5).

## 1. Mengapa perbandingan ini berbeda dari v1

Di v1, ASCA dan dense dijalankan lewat PyTorch. Yang terukur adalah framework:
~2.900 kernel launch per token decode, dispatcher, dan GEMM kecil. Perbandingan
arsitektur di dalam framework seperti itu tidak bisa dipercaya.

Di v2 keduanya ada di crate yang sama (`rust/`), memakai:

- `Mat` dan matmul yang sama
- `Param`, `AdamW` (decoupled weight decay), `RMSNorm` yang sama
- `Rng` dan urutan batch yang sama
- loop training yang sama (`bench::run`), optimizer step yang sama
- data byte-level yang sama, vocab 256, kepala output yang sama
- bahasa, compiler, dan flag optimasi yang sama

Yang berbeda **hanya arsitekturnya**. Ini satu-satunya cara klaim efisiensi jadi sah.

## 2. Protokol

- Data: TinyStories byte-level, 19,4 MB (18,5 MB train, 5% validasi di akhir)
- 400 langkah × batch 4 × seq 128 = **204.800 token** per run
- 3 seed (0,1,2), pasangan ASCA↔dense memakai seed data yang sama
- LR 3e-3 dan 6e-3, cosine decay + warmup 40, dijalankan untuk **kedua** arsitektur
- Dua mode pencocokan parameter, karena keduanya sah dan menjawab pertanyaan berbeda:
  - **total params sama** (331k): uji "kapasitas penyimpanan sama"
  - **active params sama** (99k): uji "komputasi per token sama"

## 3. Hasil

### lr 3e-3

| mode | ASCA val_loss | dense val_loss | Δ (asca−dense) | t (df=2) | tok/s ASCA | tok/s dense |
|---|---|---|---|---|---|---|
| total params 331k | **1,7925 ± 0,1005** | 2,0270 ± 0,0720 | **−0,2346 ± 0,0360** | −11,28 | 6.842 | 4.260 |
| active params 99k | **1,7925 ± 0,1005** | 2,0515 ± 0,0766 | **−0,2591 ± 0,0239** | −18,75 | 6.932 | 13.991 |

### lr 6e-3

| mode | ASCA val_loss | dense val_loss | Δ (asca−dense) | t (df=2) | tok/s ASCA | tok/s dense |
|---|---|---|---|---|---|---|
| total params 331k | **1,7343 ± 0,1229** | 1,9762 ± 0,0827 | **−0,2419 ± 0,0741** | −5,65 | 6.810 | 4.243 |
| active params 99k | **1,7343 ± 0,1229** | 1,9778 ± 0,1176 | **−0,2435 ± 0,0097** | −43,35 | 7.019 | 14.529 |

State decode pada konteks 128: **ASCA 2.048 float (konstan)** vs **dense 32.768 float
(tumbuh O(T))**. Pada konteks 1024 dense butuh 262.144 float, ASCA tetap 2.048.

### Bacaan

- Pada **total params sama**: ASCA menang kualitas ~0,24 dan **1,6× lebih cepat**,
  dengan state decode 16× lebih kecil dan tidak tumbuh terhadap konteks.
- Pada **active params sama**: ASCA masih menang kualitas ~0,24, tapi **2× lebih
  lambat** — 99k parameter dense jauh lebih murah per token daripada 331k parameter
  ASCA yang hanya 99k aktif (hash-MoE tetap harus menyimpan semua expert).
- Kedua LR memberi kesimpulan sama, jadi hasilnya tidak bergantung pada satu titik LR.

## 4. Bug yang hampir membuat klaim palsu

Percobaan pertama memberi Δ −0,35 (lebih besar). Setelah diperiksa, baseline dense
punya bug: `rope_apply` hanya merotasi slice head pertama (`row[i]` tanpa offset
`hh*dh`), sehingga **3 dari 4 head tidak punya informasi posisi sama sekali**.
Setelah diperbaiki, Δ turun ke −0,23.

Pelajaran: kalau baseline kalah, **periksa baseline dulu**, bukan arsitektur baru.
Angka yang menguntungkan diri sendiri harus dicurigai lebih dulu.

## 5. Yang BELUM membuktikan apa pun

1. **Skala sangat kecil**: 331k parameter, 205k token, konteks 128. Klaim arsitektur
   pada skala ini hampir tidak berarti; v1 sudah pernah menipu saya persis di sini
   (hasil sintetis bilang menang, teks nyata bilang kalah).
2. **Baseline belum di-tuning habis-habisan**: LR sudah di-sweep 2 titik, tapi
   inisialisasi, warmup, skala kepala, dan bobot RMSNorm belum. Dense bisa saja masih
   di bawah potensinya.
3. **Belum ada kurva loss vs token**: belum tahu apakah dense menyusul pada 1M–10M
   token. Ini uji paling penting berikutnya.
4. **Belum ada kernel CUDA**: 6.800 tok/s itu CPU dengan matmul naif. Angka tok/s
   absolut tidak bisa dibandingkan dengan apa pun di luar crate ini.
5. **Belum diuji pada task yang butuh recall jarak jauh** — di situ attention
   biasanya menang dan state terkompresi kalah.
6. **Hanya 3 seed.** Untuk klaim kuat butuh ≥5 seed dan noise floor terukur.

## 6. Langkah berikutnya (berurutan)

1. Kurva loss vs token 205k → 2M untuk kedua arsitektur, 3 seed.
2. Task recall jarak jauh (copy/associative recall sintetis) sebagai uji batas.
3. Kernel CUDA untuk delta block + matmul, lalu ukur ulang tok/s.
4. Setelah itu baru boleh bicara keunggulan.
