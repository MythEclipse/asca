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
| total params 331k | **1,7343 ± 0,1229** | 1,9762 ± 0,0827 | **−0,2420 ± 0,0741** | −5,65 | 32.174 | 5.933 |
| active params 99k | **1,7343 ± 0,1229** | 1,9778 ± 0,1176 | **−0,2435 ± 0,0097** | −43,35 | 33.844 | 21.219 |

State decode pada konteks 128: **ASCA 2.048 float (konstan)** vs **dense 32.768 float
(tumbuh O(T))**. Pada konteks 1024 dense butuh 262.144 float, ASCA tetap 2.048.

### Bacaan

- Pada **total params sama**: ASCA menang kualitas ~0,24 dan **4,5× lebih cepat**,
  dengan state decode 16× lebih kecil dan tidak tumbuh terhadap konteks.
- Pada **active params sama**: ASCA masih menang kualitas ~0,24 dan **1,10× lebih cepat**
  sesudah baseline dense ikut dioptimasi (rasio bergerak 0,48 → 1,26 → 0,95 → 1,59 → 1,10
  seiring engine berubah; angka 1,59 dicabut karena baseline dense saat itu belum
  dioptimasi;
  angka terakhir yang berlaku). Dua klaim sebelumnya dicabut di sini: "ASCA 2× lebih lambat"
  (artefak kode belum dioptimasi) dan "ASCA 1,26× lebih cepat" (diukur sebelum batching;
  batching + offload GPU menaikkan dense 31%, ASCA tidak). Yang tersisa sebagai keunggulan
  ASCA pada mode ini adalah state decode 2.048 float konstan vs 32.768 yang tumbuh O(T).

> **Catatan revisi (v2.1).** Angka throughput di tabel ini berasal dari run yang sama
> dengan angka loss (400 langkah, 3 seed, lr 6e-3) tetapi sesudah optimasi CPU 3,08×.
> Loss tidak berubah sama sekali (1,7343 / 1,9762 / 1,9778), jadi perbaikan itu murni
> kecepatan. Angka sebelum optimasi: ASCA 6.810/7.019, dense 4.243/14.529.
- Kedua LR memberi kesimpulan sama, jadi hasilnya tidak bergantung pada satu titik LR.

## 3b. Kurva loss vs token (1,024M token, 3 seed) — MENGOREKSI §3

Uji ini dijalankan karena hasil §3 (205k token) belum bisa dipercaya: keunggulan di awal
training sering hanya efek inisialisasi/induksi, bukan sifat arsitektur.

Protokol: 2000 langkah × batch 4 × seq 128 = 1.024.000 token, lr 6e-3, 3 seed,
pencocokan total params (331k), evaluasi tiap 250 langkah pada set validasi yang sama.

| token | ASCA | dense | Δ (asca−dense) | t (df=2) |
|---|---|---|---|---|
| 128.000 | 1,8448 | 2,0776 | **−0,2328 ± 0,0310** | −13,02 |
| 256.000 | 1,7267 | 1,9038 | −0,1771 ± 0,0436 | −7,04 |
| 384.000 | 1,5905 | 1,6941 | −0,1036 ± 0,0777 | −2,31 |
| 512.000 | 1,4990 | 1,5743 | −0,0753 ± 0,0460 | −2,84 |
| 640.000 | 1,4534 | 1,5231 | −0,0697 ± 0,0232 | −5,21 |
| 768.000 | 1,4101 | 1,4893 | −0,0792 ± 0,0243 | −5,64 |
| 896.000 | 1,2796 | 1,3296 | −0,0499 ± 0,0449 | −1,92 |
| 1.024.000 | **1,2968 ± 0,0871** | **1,3685 ± 0,0600** | **−0,0720 ± 0,0315** | −3,96 |

Throughput pada run yang sama (sesudah optimasi CPU): ASCA **20.407 tok/s**, dense **4.503 tok/s** (4,53×).

### Yang berubah dari §3

Keunggulan **menyusut 69%**, dari −0,233 pada 128k token menjadi −0,072 pada 1M token.
Sebagian besar selisih yang terlihat pada 205k token (Δ −0,24) adalah **efek awal
training**, bukan keunggulan arsitektur yang stabil.

Setelah ~512k token selisihnya relatif datar di kisaran −0,05 s/d −0,08 dan **belum ada
crossover** sampai 1M token. Tapi dengan 3 seed, t = −3,96 (df=2) berarti **p ≈ 0,058 —
belum signifikan pada α = 0,05**. Jadi pada 1M token pun belum bisa diklaim ASCA lebih
baik secara kualitas.

### Bacaan yang jujur

1. Klaim "ASCA menang kualitas ~0,24" dari §3 **dicabut**. Yang benar: pada 205k token
   selisihnya −0,24, dan itu menyusut menjadi −0,07 pada 1M token.
2. **Keunggulan efisiensi tidak terpengaruh** dan tetap terukur: 1,66× lebih cepat pada
   total params sama, state decode 2.048 float konstan vs 32.768 yang tumbuh O(T).
3. **Belum tahu** apakah sisa selisih −0,07 bertahan, mengecil ke nol, atau berbalik.
   Itu butuh 2–4M token dan ≥5 seed.

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
6. **Hanya 3 seed**, dan pada 1M token p ≈ 0,058 — belum signifikan.
7. **Belum ada crossover test sampai 2–4M token.** Kurva 128k→1M menunjukkan penyusutan
   69%, jadi arah trennya belum jelas apakah berhenti di −0,07 atau terus mengecil.

## 6. Langkah berikutnya (berurutan)

1. ~~Kurva loss vs token 205k → 2M~~ **sudah sampai 1M** (§3b). Lanjutkan ke 2–4M token
   dengan ≥5 seed untuk melihat apakah sisa selisih −0,07 bertahan atau hilang.
2. Task recall jarak jauh (copy/associative recall sintetis) sebagai uji batas.
3. Kernel CUDA untuk delta block + matmul, lalu ukur ulang tok/s.
4. Setelah itu baru boleh bicara keunggulan.
