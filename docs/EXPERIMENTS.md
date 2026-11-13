> **ARSIP v1.** Log eksperimen ini berasal dari implementasi PyTorch yang sudah dihapus
> dari repo (semua kode `asca/`, `baseline/`, `benchmarks/`, `tests/`). Angka di bawah
> tetap valid sebagai catatan sejarah. Hasil yang berlaku sekarang ada di
> [`RESULTS_V2.md`](RESULTS_V2.md) dan [`CUDA.md`](CUDA.md).

# EXPERIMENTS — ASCA

Hasil benchmark otomatis. Setiap entri menyertakan konfigurasi lengkap.

## Cara membaca hasil ini

- Semua angka dihasilkan oleh skrip di `benchmarks/` pada **RTX 4050 Laptop 6 GB**,
  torch `2.11.0+cu128`, bf16 autocast.
- `active_params` = parameter yang benar-benar dipakai per token (MoE dihitung top-k saja).
- `flops_per_token` **diukur** dengan `torch.utils.flop_counter` (matmul/conv yang
  benar-benar dieksekusi), bukan estimasi rumus.
- Klaim hanya boleh dibuat dari **mean ± std minimal 3 seed**. Satu run = anekdot.
- `B0-dense-matched` = dense transformer yang `d_ff`-nya dicari otomatis agar jumlah
  active params setara ASCA. Ini pembanding yang adil untuk MoE.
- **Noise floor terukur**: konfigurasi `A6` identik dengan `A0` (default
  `router_threshold=0.0`), sehingga selisih val loss antara keduanya pada run
  ablasi (0.559 vs 0.638, 300 step) murni berasal dari inisialisasi/urutan batch.
  **Selisih < ~0.08 val loss pada budget ini tidak bermakna**; jangan menafsirkan
  satu seed sebagai bukti.

## Diagnosis defisit: LR sweep + kapasitas (TinyStories 12M, 3 seed)

Dua hipotesis diuji: (a) defisit karena **optimisasi** (LR tidak di-tuning per
arsitektur), (b) defisit karena **kapasitas** (MoE kekurangan kapasitas efektif).
Protokol: TinyStories byte-level, 12M token, seq 1024, batch 8, 3 seed, `torch.compile`.

| Model | lr 1e-3 | lr 3e-3 | lr 6e-3 | lr 1e-2 |
|---|---|---|---|---|
| dense-matched | 0.9319 ± 0.0123 | 0.9013 ± 0.0160 | **0.8875 ± 0.0079** | 0.8967 ± 0.0218 |
| ASCA (8 expert) | 1.0481 ± 0.0069 | 0.9246 ± 0.0099 | **0.8973 ± 0.0105** | 0.8977 ± 0.0138 |
| **Δ (ASCA − dense)** | **+0.1162** | **+0.0234** | **+0.0098** | **+0.0011** |

Kapasitas (lr 3e-3, top_k tetap 2 sehingga active params ~sama):

| n_experts | val loss | throughput | VRAM |
|---|---|---|---|
| 4 | 0.9310 ± 0.0086 | 220k tok/s | 627 MB |
| 8 | 0.9246 ± 0.0099 | 231k tok/s | 718 MB |
| 16 | 0.9078 ± 0.0104 | 196k tok/s | 864 MB |

Kombinasi terbaik: **16 expert + lr 6e-3 → 0.8813 ± 0.0121** (210k tok/s, 967 MB).

### Koreksi: hasil "ASCA kalah signifikan (p=0.042)" sebagian besar artefak LR

Protokol lama memakai **satu LR (3e-3) untuk kedua arsitektur**. Ternyata itu
merugikan ASCA: pada lr 3e-3 selisihnya +0.0234, tetapi

- pada lr 1e-2 selisihnya tinggal **+0.0011** (praktis nol),
- pada LR optimal masing-masing (6e-3 untuk keduanya) selisihnya **+0.0098** ≈ 1σ,
- ASCA 16 expert pada lr 6e-3 malah **−0.0062 di bawah** dense (dalam noise).

Jadi klaim "ASCA secara signifikan lebih buruk pada teks nyata" **dicabut**. Yang
tersisa: **tidak ada perbedaan kualitas yang bisa dideteksi** pada skala ini, ke dua
arah. Uji p=0.042 sebelumnya sah secara statistik untuk konfigurasi yang diuji, tetapi
konfigurasi itu tidak adil untuk ASCA.

### Yang tetap berlaku

1. **Tidak ada keunggulan kualitas.** Hasil terbaik ASCA (0.8813) dan dense (0.8875)
   berada dalam 1σ; dengan 3 seed ini tidak bisa dipisahkan. Butuh ≥5 seed.
2. **Defisit efisiensi justru makin besar.** ASCA 3.7× lebih lambat (210k vs 756k
   tok/s) dan 3.4× lebih boros VRAM (967 vs 286 MB) pada titik kualitas terbaiknya.
   Untuk arsitektur yang menjual efisiensi, ini masalah utama — bukan kualitas.
3. **Kapasitas memang menolong ASCA** (4→8→16 expert monoton membaik), jadi sebagian
   defisit lama berasal dari kapasitas efektif yang terlalu kecil, bukan hanya LR.
4. **Aturan protokol baru**: LR harus di-tuning per arsitektur sebelum klaim apa pun.
   Perbandingan dengan satu LR bersama tidak valid.

## ⚠️ Korpus NYATA (TinyStories): hasil sintetis TIDAK terprediksi

Ini eksperimen paling penting di dokumen ini. Semua eksperimen sebelumnya memakai
korpus **sintetis** (48 kata acak, struktur statistik dangkal). Di sini dipakai teks
nyata: TinyStories (19.0M token, tokenizer byte-level, vocab 241), `seq_len=1024`,
12M token latih, 3 seed, `dense-matched` sebagai kontrol.

| Model | val loss | vs dense (paired) | throughput | VRAM |
|---|---|---|---|---|
| **dense-matched** | **0.8998 ± 0.0150** | — | **797k tok/s** | 286 MB |
| A0 (penuh) | 0.9240 ± 0.0081 | **+0.0242** (t=1.90) | 225k tok/s | 715 MB |
| A2 (tanpa cabang linear) | 0.9309 ± 0.0031 | +0.0311 (t=3.06) | 331k tok/s | 536 MB |
| A4 (tanpa memory) | 0.9227 ± 0.0064 | +0.0229 (t=4.61) | 246k tok/s | 613 MB |
| A8 (ramping) | 0.9899 ± 0.0177 | **+0.0901** (t=4.82) | 366k tok/s | 488 MB |

Semua tanda positif = **lebih buruk** dari dense.

### Tiga temuan yang membalik kesimpulan sebelumnya

1. **Keunggulan di korpus sintetis tidak muncul di teks nyata.** Di sintetis, A0 vs
   dense = −0.0023 (seq 256) sampai −0.0091 (seq 1024); di teks nyata A0 vs dense =
   **+0.0242 (lebih buruk)**. Arah tandanya berbalik.
2. **Mekanisme yang tampak "beban mati" di sintetis ternyata berguna di teks nyata.**
   A8 (tanpa semantic encoder + cabang linear + shared expert) hanya −0.0018 dari A0
   di sintetis, tapi **+0.0901 lebih buruk dari dense** di teks nyata — degradasi 4×
   lebih besar daripada melepas komponen lain. Jadi komponen itu memang membawa
   sinyal nyata; kesimpulan "bisa dibuang" adalah artefak data sintetis.
3. **Pada skala ini ASCA tidak mengalahkan dense pada teks nyata.** A4 (t=4.6) dan A8
   (t=4.8) jelas lebih buruk; A0 lebih buruk secara arah tapi baru marginal
   (t=1.90, df=2, p≈0.15) — perlu seed tambahan untuk memastikan.

### Konsekuensi metodologis (aturan baru proyek)

> **Ablasi pada korpus sintetis tidak boleh dipakai untuk keputusan arsitektur.**
> Semua angka sintetis di dokumen ini tetap disimpan sebagai catatan historis, tetapi
> kesimpulan arsitektural hanya boleh diambil dari korpus nyata.

Perbedaan varian antar-seed juga jauh lebih besar di teks nyata (±0.015 untuk dense,
±0.008 untuk A0) daripada di sintetis (±0.0006). Artinya **noise floor bergantung
pada korpus**: pada teks nyata, delta < ~0.02 belum bisa dipercaya dengan 3 seed.

### Konfirmasi 5 seed, 24M token (TinyStories, seq 1024)

| Model | val loss | throughput | VRAM |
|---|---|---|---|
| **dense-matched** | **0.8065 ± 0.0105** | 769k tok/s | 286 MB |
| ASCA (A0) | 0.8282 ± 0.0069 | 224k tok/s | 709 MB |

Paired per seed (A0 − dense): +0.0095, +0.0345, +0.0441, +0.0091, +0.0112 →
**+0.0217 ± 0.0074, t = 2.95, df = 4, p = 0.042**. Kelima seed bertanda positif.

**Kesimpulan yang boleh diambil: pada teks nyata, skala ini, protokol ini, ASCA
secara signifikan LEBIH BURUK dari dense-matched** (p < 0.05) — dan 3.4× lebih
lambat dengan 2.5× VRAM lebih besar. Ini hasil negatif yang jelas, bukan noise.

### Kenapa ASCA kalah di teks nyata (hipotesis, belum diuji)

- Kapasitas efektif lebih kecil: `d_model=128` sama, tapi active params ASCA
  (~0.9M) tersebar ke 4 sublayer + 8 expert, sedangkan `dense-matched` menaruh
  semuanya di satu MLP lebar. Pada teks nyata (yang butuh kapasitas model bahasa),
  lebar MLP lebih berharga daripada mekanisme struktural.
- Byte-level tokenizer + 12M token = rezim sangat rendah data; arsitektur yang lebih
  sederhana lebih mudah dioptimisasi.
- Belum ada tuning LR/regularisasi per arsitektur (protokol sama untuk semua).

## Konteks panjang: seq 1024 (20M token, nano, 3 seed)

Protokol: 2442 step × 8 × 1024 = 20.0M token. `local_window = 128`,
`mem_segment_len = 256` → **4 segmen nyata**, jadi modul memori dan cabang linear
benar-benar bekerja (di seq 256 keduanya hampir menganggur).

| Model | val loss | throughput | VRAM | vs dense |
|---|---|---|---|---|
| dense-matched | 0.3888 ± 0.0006 | **810k tok/s** | 282 MB | — |
| A0 (penuh) | 0.3797 ± 0.0037 | 218k tok/s | 689 MB | −0.0091 (t=4.2, df≈2) |
| A2 tanpa cabang linear | **0.3782 ± 0.0042** | 327k tok/s | 519 MB | −0.0106 |
| A4 tanpa memory | 0.3835 ± 0.0007 | 235k tok/s | 601 MB | −0.0053 (t=9.9) |
| A8 (ramping) | 0.3808 ± 0.0069 | 368k tok/s | 482 MB | −0.0080 |

### Sinyal pertama yang melewati noise floor — tapi belum konklusif

Pada seq 1024, selisih ASCA vs dense (−0.0091) **melewati** noise floor yang diukur
(±0.005). Tapi:

- Welch t = 4.17 dengan **df ≈ 2** → p ≈ 0.05, marginal.
- Sebaran antar-run untuk A0 sendiri diukur 0.005–0.007 (seed 0 di dua proses
  berbeda: 0.3966 vs 0.4018 pada seq 256). Delta 0.0091 hanya ~1.5× sebaran itu.
- Hanya 3 seed, satu dataset sintetis, satu ukuran model.

**Status: hipotesis, bukan klaim.** Perlu ≥5 seed dan dataset nyata.

### Tren yang layak diuji (belum dikonfirmasi)

| Panjang konteks | ASCA (A0) | dense-matched | Δ |
|---|---|---|---|
| 256 | 0.3946 ± 0.0063 | 0.3968 ± 0.0053 | −0.0023 |
| 1024 | 0.3797 ± 0.0037 | 0.3888 ± 0.0006 | −0.0091 |

Delta membesar ~4× saat konteks naik 4×, sementara dense hampir tidak membaik
(0.3968 → 0.3888) dan ASCA membaik jauh lebih banyak (0.3946 → 0.3797). Konsisten
dengan dugaan bahwa mekanisme memori/linear mulai berguna pada konteks panjang.
**Caveat berat: run seq 256 memakai batch 32, seq 1024 memakai batch 8** — batch size
memengaruhi optimisasi, jadi perbandingan lintas-seq ini terkontaminasi.

### Temuan yang konsisten di kedua panjang konteks

1. **Cabang linear attention tetap tidak membayar biayanya**, bahkan pada seq 1024
   dengan window 128 (di mana ia seharusnya paling berguna): A2 = A0 secara kualitas
   (0.3782 vs 0.3797) tapi **1.5× lebih cepat dan 25% lebih hemat VRAM**.
2. **Varian ramping A8 tetap setara kualitas** dengan A0 (0.3808 vs 0.3797) pada
   **1.7× throughput** dan 30% VRAM lebih rendah.
3. **Biaya ASCA tumbuh lebih cepat terhadap konteks daripada dense**: throughput A0
   turun 388k → 218k saat seq 256 → 1024, sementara dense tetap ~810k. Gap melebar
   dari 2.4× ke 3.7×. Ini masalah arsitektural (sublayer tambahan + biaya per segmen),
   bukan masalah kernel.
4. **Memori: arah positif tapi lemah.** A4 (tanpa memory) lebih buruk 0.0038 dari A0
   (~1σ) — kali ini dengan 4 segmen nyata, berbeda dari seq 256 yang tidak valid.
   Belum cukup untuk diklaim.

## Ablasi A0–A9 pada 20M token (nano, batch 32, seq 256, 3 seed)

Protokol: 2442 step × 32 × 256 = **20.0M token** per run, LR 3e-3 cosine + 100 warmup,
`torch.compile`, evaluasi 20 batch. `dense-matched` = dense baseline dengan active
params disetarakan (binary search `d_ff`). Semua run memakai seed 0,1,2 kecuali yang
ditandai 1 seed.

| Model | val loss | throughput | peak VRAM | catatan |
|---|---|---|---|---|
| dense-matched (kontrol) | 0.3968 ± 0.0053 | **914k tok/s** | 282 MB | baseline |
| **A0 (ASCA penuh)** | 0.3946 ± 0.0063 | 388k tok/s | 584 MB | Δ vs dense −0.0023 (**< 0.5σ**) |
| A1 tanpa semantic encoder | 0.3909 (1 seed) | 415k tok/s | 595 MB | |
| A2 tanpa cabang linear | 0.3908 (1 seed) | 547k tok/s | 422 MB | |
| A3 tanpa MoE | 0.3961 ± 0.0006 | 558k tok/s | 419 MB | |
| A4 tanpa memory | 0.3964 ± 0.0001 | 382k tok/s | 586 MB | |
| A5 tanpa shared expert | 0.3968 (1 seed) | 450k tok/s | 586 MB | |
| A6 (identik A0) | 0.3962 (1 seed) | 412k tok/s | 594 MB | **kontrol noise** |
| A7 = A1+A2 | 0.3961 ± 0.0057 | 512k tok/s | 420 MB | |
| **A8 = A7 tanpa shared expert** | 0.3928 ± 0.0033 | **605k tok/s** | **389 MB** | 1.56× lebih cepat dari A0 |
| A9 = A8 tanpa memory | 0.3972 ± 0.0061 | 593k tok/s | 391 MB | |

### Noise floor (diukur, bukan diasumsikan)

- Std antar-seed dalam satu config: **0.0001 – 0.0063** (median ≈ 0.005).
- Sebaran proses-ke-proses untuk config & seed yang sama (A0 seed 0 di dua proses):
  **0.0052** (0.3966 vs 0.4018).
- **Kesimpulan: delta val loss < ~0.006 TIDAK bermakna pada protokol ini.** Semua
  selisih kualitas di tabel atas berada di bawah atau sekitar ambang itu.

### Yang bisa dan tidak bisa disimpulkan

**TIDAK bisa disimpulkan (semua di bawah noise):**

1. ASCA lebih baik dari dense-matched: Δ = −0.0023 (< 0.5σ). **Tidak ada keunggulan.**
2. Komponen mana pun memperbaiki kualitas: melepas semantic encoder, cabang linear,
   MoE, memory, atau shared expert semuanya menghasilkan val loss dalam ±0.005.
3. MoE/memory "membayar biayanya" secara kualitas — data 1 seed sebelumnya tampak
   begitu, tapi hilang setelah 3 seed.

**BISA disimpulkan (efeknya besar dan deterministik):**

1. **Biaya komponen sangat tidak seimbang.** A8 (tanpa semantic encoder, tanpa cabang
   linear, tanpa shared expert) mencapai kualitas yang sama dengan **1.56× throughput
   lebih tinggi dan 1.5× VRAM lebih rendah** daripada A0.
2. **A8 hanya 1.51× lebih lambat dari dense-matched** (605k vs 914k tok/s), versus
   A0 yang 2.36× lebih lambat. Gap throughput training sebagian besar berasal dari
   komponen yang tidak memberi kualitas terukur.
3. **Cabang linear attention adalah biaya terbesar**: melepasnya (A2) menaikkan
   throughput 41% (388k → 547k) dan memangkas VRAM 28% (584 → 422 MB).

**Rekomendasi berbasis data:** jadikan **A8 sebagai konfigurasi default** untuk
eksperimen berikutnya; mekanisme yang dilepas bisa diuji ulang nanti pada skala/seq
yang lebih panjang, di mana ia mungkin baru mulai berguna.

### Caveat penting

- `seq_len = 256` dengan `mem_segment_len = 256` berarti modul memori hampir tidak
  punya pekerjaan (hanya 1 batas segmen). **Ablasi A4/A9 belum menguji memori secara
  adil**; perlu seq ≥ 1024 atau `mem_segment_len` lebih kecil.
- Semua run hanya 20M token dan `d_model=128`. Ini rezim "mainan": klaim skala besar
  tetap tidak boleh dibuat.
- Tidak ada tuning hyperparameter per model; LR/batch sama untuk semua.

## Status ringkas v0.8 — koreksi hasil negatif: defisit kualitas sebagian besar artefak LR

**Koreksi penting.** Klaim v0.7 ("ASCA signifikan lebih buruk, p=0.042") **dicabut**.
Itu diukur dengan satu LR (3e-3) untuk kedua arsitektur. Setelah LR di-tuning per
arsitektur pada korpus yang sama (TinyStories, 12M token, 3 seed):

| lr | dense-matched | ASCA (8 expert) | Δ |
|---|---|---|---|
| 1e-3 | 0.9319 | 1.0481 | +0.1162 |
| 3e-3 | 0.9013 | 0.9246 | +0.0234 |
| 6e-3 | **0.8875** | **0.8973** | **+0.0098** |
| 1e-2 | 0.8967 | 0.8977 | **+0.0011** |

ASCA 16 expert @ lr 6e-3: **0.8813** (nominal terbaik, dalam noise vs dense).

**Status klaim sekarang: tidak ada perbedaan kualitas yang terdeteksi, ke dua arah.**
Bukan "ASCA lebih buruk", juga bukan "ASCA lebih baik". Yang jelas dan terukur
hanyalah **biaya**: pada titik kualitas terbaiknya ASCA 3.7× lebih lambat dan 3.4×
lebih boros VRAM. Untuk arsitektur yang menjual efisiensi, itu masalah utamanya.

## Status ringkas v0.7 — hasil NEGATIF pada teks nyata (baca ini dulu)

**Pada teks nyata, ASCA kalah dari baseline dense.** TinyStories (byte-level),
24M token, seq 1024, 5 seed:

| | val loss | throughput | VRAM |
|---|---|---|---|
| dense-matched | **0.8065 ± 0.0105** | **769k tok/s** | **286 MB** |
| ASCA penuh | 0.8282 ± 0.0069 | 224k tok/s | 709 MB |

Paired t = 2.95, df = 4, **p = 0.042** — kelima seed lebih buruk. Bukan noise.

**Dan yang lebih penting: hasil pada korpus sintetis tidak terprediksi.** Di data
sintetis ASCA tampak sedikit *lebih baik* (Δ −0.002…−0.009) dan varian rampingnya
tampak "sama saja"; di teks nyata tandanya berbalik dan varian ramping jadi jauh
lebih buruk (+0.09). **Ablasi sintetis tidak dipakai lagi untuk keputusan arsitektur.**

Status klaim: **tidak ada keunggulan apa pun.** Yang tersisa hanyalah keunggulan
memori cache decode (3.8 MB konstan vs 24 MB) yang tidak cukup menutupi defisit
kualitas dan throughput.

## Status ringkas v0.6 — hasil ablation 20M token (baca ini dulu)

**Kesimpulan utama: pada 20M token / nano, tidak ada konfigurasi yang lebih baik
secara kualitas, dan tidak ada komponen yang terbukti menyumbang kualitas.** Yang
terbukti hanyalah biaya: komponen terberat tidak menghasilkan apa pun yang terukur,
dan melepasnya membuat ASCA 1.56× lebih cepat tanpa kehilangan kualitas.

| | val loss | throughput | VRAM |
|---|---|---|---|
| dense-matched | 0.3968 ± 0.0053 | 914k tok/s | 282 MB |
| ASCA penuh (A0) | 0.3946 ± 0.0063 | 388k tok/s | 584 MB |
| **ASCA ramping (A8)** | **0.3928 ± 0.0033** | **605k tok/s** | **389 MB** |

Noise floor terukur: **±0.005** (seed & proses). Semua delta kualitas di proyek ini
sejauh ini berada di bawah ambang itu → **tidak ada klaim keunggulan kualitas.**

Lihat bagian **“Ablasi A0–A9”** di bawah untuk tabel lengkap, analisis noise, dan
caveat (terutama: ablasi memori belum valid karena `seq_len == mem_segment_len`).

## Status ringkas v0.5 (prefill paralel + CUDA Graph, tiny)

| Aspek | ASCA | dense-matched | Rasio |
|---|---|---|---|
| **Prefill paralel** (tok/s, konteks 2036) | **48.955** | 64.690 | **1.32× lebih lambat** |
| Prefill token-demi-token (v0.4) | 611 | 1.543 | 2.5× lebih lambat |
| Decode CUDA Graph (ms/tok) | **1.60** | 0.65 | 2.5× lebih lambat |
| Decode eager (ms/tok) | 8.58 | 2.20 | 3.9× |
| Memori cache decode | **3.82 MB konstan** | 24.25 MB | 6.3× lebih kecil |
| val loss (nano, 3 seed) | 0.4159 ± 0.0020 | 0.4210 ± 0.0145 | setara |
| Throughput training (nano) | 103.9k tok/s | 557k tok/s | 5.4× lebih lambat |

**Prefill paralel.** `ASCAModel.prefill(idx)` memproses prompt lewat forward chunked
sekaligus, lalu **menurunkan state cache decode dari lintasan paralel yang sama**:
ring buffer window (posisi `T-w..T-1` pada slot `pos % w`), state linear akhir dari
loop chunk, ringkasan segmen yang sudah selesai (`T // L_s`), akumulator online-softmax
untuk segmen aktif, dan buffer konvolusi encoder. Tidak ada token yang diproses dua
kali dan tidak ada panggilan `forward_step` per token.

Diuji identik dengan loop `forward_step` T kali (galat ~1e-7) di
`test_prefill_matches_step_loop` dan `test_dense_prefill_matches_step_loop`.

**Yang masih gagal (jujur):**

1. **Decode masih 2.5× lebih lambat** dari dense (1.60 vs 0.65 ms/tok). GPU-side
   work ASCA memang lebih banyak: 4 sublayer per blok + state linear O(Dh²) per head.
2. **Prefill 1.32× lebih lambat** pada konteks panjang (dan 2.7× pada konteks 512,
   karena overhead lebih dominan di sana). Ini pertama kalinya ASCA mendekati
   parity di salah satu metrik — tapi belum menang.
3. **Throughput & VRAM training tidak berubah**: 5.4× lebih lambat, 1.9× VRAM.
   Itu bottleneck terbesar yang tersisa.
4. **Belum ada benchmark skala besar** (≥100M token), semua angka di sini dari
   nano/tiny dan 1.23M token. **Tidak ada klaim keunggulan.**

## Training komparasi (nano, 150 step)

_2026-11-12 18:28:09 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 150,
  "batch": 8,
  "seq_len": 256,
  "train_tokens": 307200,
  "lr": 0.003,
  "warmup": 10,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 256,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense",
    "dense-matched",
    "asca"
  ]
}
```

| model | params | active_params | flops_per_token | tokens | steps | train_loss | val_loss | ppl | secs | tokens_per_sec | peak_vram_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-dense | 807040 | 791680 | 2136064 | 307200 | 150 | 0.9401 | 0.9349 | 2.547 | 1.3 | 235861.8 | 98.8 |
| B0-dense-matched | 1224832 | 1209472 | 2971648 | 307200 | 150 | 0.8826 | 0.8727 | 2.393 | 0.93 | 331499.4 | 124.1 |
| A0-asca | 1818389 | 1213205 | 2613312 | 307200 | 150 | 0.8939 | 0.8967 | 2.451 | 5.02 | 61195.7 | 175.1 |


## Training komparasi (nano, 600 step — 3 seed, 1.23M token)

_2026-11-12 18:29:50 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 600,
  "batch": 8,
  "seq_len": 256,
  "train_tokens": 1228800,
  "lr": 0.003,
  "warmup": 10,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 256,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca"
  ]
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1224832 | 1209472 | 2971648 | 1228800 | 0.4093 ± 0.0018 | 1.506 ± 0.003 | 344209 ± 17905 | 124 ± 0 | 3 |
| A0-asca | 1818389 | 1213205 | 2613312 | 1228800 | 0.4144 ± 0.0034 | 1.514 ± 0.005 | 63001 ± 283 | 175 ± 0 | 3 |


## VRAM & latensi decode (nano)

_2026-11-12 18:30:02 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "seq_lens": [
    512,
    1024,
    2048,
    4096
  ],
  "vocab": 4096,
  "seed": 0,
  "device": "cuda"
}
```

| model | params | train_vram_T512 | train_vram_T1024 | train_vram_T2048 | train_vram_T4096 | decode_ms_per_tok |
|---|---|---|---|---|---|---|
| B0-dense | 1315968 | 61.6 | 120.6 | 211.1 | 391.9 | 1.38 |
| ASCA-full | 2327317 | 102.1 | 187.9 | 360.4 | 782.9 | 10.37 |
| B0-dense-matched | 1733760 | 100.8 | 157.5 | 256.8 | 455.5 | 1.36 |


## Training komparasi (nano, 300 step — ablasi awal)

_2026-11-12 18:31:51 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 300,
  "batch": 8,
  "seq_len": 256,
  "train_tokens": 614400,
  "lr": 0.003,
  "warmup": 10,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 256,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "asca",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6"
  ]
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| A0-asca | 1818389 | 1213205 | 2613312 | 614400 | 0.5590 ± 0.0000 | 1.749 ± 0.000 | 60983 ± 0 | 175 ± 0 | 1 |
| ASCA-A1 | 1809557 | 1204373 | 2595904 | 614400 | 0.6454 ± 0.0000 | 1.907 ± 0.000 | 64396 ± 0 | 172 ± 0 | 1 |
| ASCA-A2 | 1750277 | 1145093 | 2284544 | 614400 | 0.5156 ± 0.0000 | 1.675 ± 0.000 | 93241 ± 0 | 141 ± 0 | 1 |
| ASCA-A3 | 1163029 | 1147669 | 2482240 | 614400 | 0.6432 ± 0.0000 | 1.903 ± 0.000 | 121778 ± 0 | 136 ± 0 | 1 |
| ASCA-A4 | 1539345 | 934161 | 2613312 | 614400 | 0.5177 ± 0.0000 | 1.678 ± 0.000 | 63792 ± 0 | 174 ± 0 | 1 |
| ASCA-A5 | 1425173 | 819989 | 1826880 | 614400 | 0.8930 ± 0.0000 | 2.442 ± 0.000 | 66406 ± 0 | 147 ± 0 | 1 |
| ASCA-A6 | 1818389 | 1213205 | 2613312 | 614400 | 0.6380 ± 0.0000 | 1.893 ± 0.000 | 63114 ± 0 | 175 ± 0 | 1 |

## Training komparasi (nano, 600 step — kernel opt (chunk adaptif + grouped GEMM + compile))

_2026-11-12 19:25:23 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 600,
  "batch": 8,
  "seq_len": 256,
  "train_tokens": 1228800,
  "lr": 0.003,
  "warmup": 10,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 256,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1224832 | 1209472 | 2971648 | 1228800 | 0.4167 ± 0.0063 | 1.517 ± 0.009 | 534077 ± 23887 | 99 ± 0 | 3 |
| A0-asca | 1818389 | 1213205 | 3651648 | 1228800 | 0.4165 ± 0.0020 | 1.517 ± 0.003 | 106060 ± 1545 | 185 ± 0 | 3 |


## FLOPs & throughput (tiny, seq 512)

_2026-11-12 19:26:39 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "tiny",
  "seq_len": 512,
  "batch": 4,
  "vocab": 4096,
  "seed": 0,
  "device": "cuda",
  "ablation": null,
  "asca_config": {
    "vocab_size": 4096,
    "d_model": 256,
    "n_layers": 6,
    "n_heads": 8,
    "head_dim": 32,
    "max_seq_len": 512,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 128,
    "shared_ff": 512,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 688,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  }
}
```

| model | params | active_params | flops_per_token | train_tokens_per_sec | peak_vram_mb | moe_pad_factor |
|---|---|---|---|---|---|---|
| B0-dense (default) | 5795072 | 4746496 | 14729216 | 203770.4 | 365.9 | None |
| B0-dense (param-matched) | 8264960 | 7216384 | 19668992 | 151673.4 | 457.6 | None |
| ASCA-full | 11779639 | 7192119 | 21110144 | 57724.3 | 760.8 | 1.07 |


## Training komparasi (nano, 600 step — setelah fix normalisasi memori + KV-cache)

_2026-11-12 19:54:57 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 600,
  "batch": 8,
  "seq_len": 256,
  "train_tokens": 1228800,
  "lr": 0.003,
  "warmup": 10,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 256,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1224832 | 1209472 | 2971648 | 1228800 | 0.4210 ± 0.0145 | 1.524 ± 0.022 | 557458 ± 31812 | 99 ± 0 | 3 |
| A0-asca | 1818389 | 1213205 | 3520512 | 1228800 | 0.4159 ± 0.0020 | 1.516 ± 0.003 | 103871 ± 5261 | 190 ± 3 | 3 |


## Training komparasi (nano, 2442 step — ABLASI 20M token, 1 seed, nano (bagian 2))

_2026-11-12 21:11:04 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 2442,
  "batch": 32,
  "seq_len": 256,
  "train_tokens": 20004864,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 256,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "A4",
    "A5",
    "A6"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| ASCA-A4 | 1538817 | 933633 | 3572736 | 20004864 | 0.3992 ± 0.0000 | 1.491 ± 0.000 | 402245 ± 0 | 592 ± 0 | 1 |
| ASCA-A5 | 1424645 | 819461 | 2795520 | 20004864 | 0.3968 ± 0.0000 | 1.487 ± 0.000 | 449520 ± 0 | 586 ± 0 | 1 |
| ASCA-A6 | 1817861 | 1212677 | 3666432 | 20004864 | 0.3962 ± 0.0000 | 1.486 ± 0.000 | 412045 ± 0 | 594 ± 0 | 1 |


## Training komparasi (nano, 2442 step — ABLASI lanjutan 20M token, 3 seed, nano)

_2026-11-12 21:20:33 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 2442,
  "batch": 32,
  "seq_len": 256,
  "train_tokens": 20004864,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 256,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca",
    "A7",
    "A8"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1224832 | 1209472 | 2971648 | 20004864 | 0.3968 ± 0.0043 | 1.487 ± 0.006 | 911371 ± 1683 | 282 ± 0 | 3 |
| A0-asca | 1817861 | 1212677 | 3648000 | 20004864 | 0.3946 ± 0.0051 | 1.484 ± 0.008 | 387992 ± 10904 | 582 ± 8 | 3 |
| ASCA-A7 | 1741445 | 1136261 | 3310080 | 20004864 | 0.3961 ± 0.0046 | 1.486 ± 0.007 | 511927 ± 6382 | 420 ± 8 | 3 |
| ASCA-A8 | 1348229 | 743045 | 2488320 | 20004864 | 0.3928 ± 0.0027 | 1.481 ± 0.004 | 604921 ± 34093 | 388 ± 2 | 3 |


## Training komparasi (nano, 2442 step — ABLASI 20M token 3 seed: MoE & memory)

_2026-11-12 21:30:36 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 2442,
  "batch": 32,
  "seq_len": 256,
  "train_tokens": 20004864,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 256,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "A3",
    "A4",
    "A9"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| ASCA-A3 | 1162501 | 1147141 | 2486272 | 20004864 | 0.3961 ± 0.0004 | 1.486 ± 0.001 | 558251 ± 2855 | 419 ± 0 | 3 |
| ASCA-A4 | 1538817 | 933633 | 3601920 | 20004864 | 0.3964 ± 0.0001 | 1.487 ± 0.000 | 382468 ± 10092 | 586 ± 8 | 3 |
| ASCA-A9 | 1069185 | 464001 | 2466816 | 20004864 | 0.3972 ± 0.0050 | 1.488 ± 0.008 | 592550 ± 10543 | 391 ± 2 | 3 |


## Training komparasi (nano, 2442 step — KONTEKS PANJANG 1024: memori & dense)

_2026-11-12 21:54:15 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 2442,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 20004864,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca",
    "A4"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1224832 | 1209472 | 4544512 | 20004864 | 0.3888 ± 0.0005 | 1.475 ± 0.001 | 810378 ± 268 | 282 ± 0 | 3 |
| A0-asca | 1817861 | 1212677 | 6002176 | 20004864 | 0.3797 ± 0.0031 | 1.462 ± 0.004 | 217804 ± 2109 | 689 ± 8 | 3 |
| ASCA-A4 | 1538817 | 933633 | 5133312 | 20004864 | 0.3835 ± 0.0005 | 1.467 ± 0.001 | 235063 ± 1855 | 602 ± 7 | 3 |


## Training komparasi (nano, 2442 step — KONTEKS PANJANG 1024: cabang linear & lean)

_2026-11-12 22:00:30 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 2442,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 20004864,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "A2",
    "A8"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| ASCA-A2 | 1750277 | 1145093 | 5673984 | 20004864 | 0.3782 ± 0.0035 | 1.460 ± 0.005 | 327408 ± 1721 | 519 ± 7 | 3 |
| ASCA-A8 | 1348229 | 743045 | 4880896 | 20004864 | 0.3808 ± 0.0056 | 1.464 ± 0.008 | 368122 ± 5859 | 482 ± 6 | 3 |


## Training komparasi (nano, 2442 step — KONTEKS PANJANG 2048)

_2026-11-12 22:16:38 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "synthetic",
  "tokens_in_corpus": 2000014,
  "vocab": 120,
  "steps": 2442,
  "batch": 4,
  "seq_len": 2048,
  "train_tokens": 20004864,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 120,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 2048,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca",
    "A8"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1224832 | 1209472 | 6641664 | 20004864 | 0.3860 ± 0.0001 | 1.471 ± 0.000 | 691819 ± 13699 | 282 ± 0 | 3 |
| A0-asca | 1817861 | 1212677 | 8353600 | 20004864 | 0.3791 ± 0.0051 | 1.461 ± 0.007 | 151185 ± 2225 | 737 ± 7 | 3 |
| ASCA-A8 | 1348229 | 743045 | 7181632 | 20004864 | 0.3777 ± 0.0062 | 1.459 ± 0.009 | 253578 ± 2302 | 539 ± 4 | 3 |


## Training komparasi (nano, 1465 step — KORPUS NYATA TinyStories seq1024)

_2026-11-12 22:27:33 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca",
    "A2"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1240320 | 1209472 | 4575488 | 12001280 | 0.8998 ± 0.0123 | 2.459 ± 0.030 | 795608 ± 1618 | 286 ± 0 | 3 |
| A0-asca | 1833349 | 1212677 | 5890304 | 12001280 | 0.9240 ± 0.0066 | 2.519 ± 0.017 | 225450 ± 4038 | 715 ± 5 | 3 |
| ASCA-A2 | 1765765 | 1145093 | 5528704 | 12001280 | 0.9309 ± 0.0025 | 2.537 ± 0.006 | 330808 ± 2312 | 536 ± 6 | 3 |


## Training komparasi (nano, 1465 step — KORPUS NYATA TinyStories seq1024 (lanjutan))

_2026-11-12 22:32:16 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "A4",
    "A8"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| ASCA-A4 | 1554305 | 933633 | 5057920 | 12001280 | 0.9227 ± 0.0052 | 2.516 ± 0.013 | 246339 ± 2355 | 613 ± 7 | 3 |
| ASCA-A8 | 1363717 | 743045 | 4749824 | 12001280 | 0.9899 ± 0.0144 | 2.691 ± 0.039 | 365960 ± 1689 | 489 ± 9 | 3 |


## Training komparasi (nano, 2929 step — KORPUS NYATA 24M token 5 seed)

_2026-11-12 22:44:49 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 2929,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 23994368,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1240320 | 1209472 | 4575488 | 23994368 | 0.8065 ± 0.0094 | 2.240 ± 0.021 | 769441 ± 15774 | 286 ± 0 | 5 |
| A0-asca | 1833349 | 1212677 | 5890304 | 23994368 | 0.8282 ± 0.0061 | 2.289 ± 0.014 | 223684 ± 1982 | 709 ± 7 | 5 |


## Training komparasi (nano, 1465 step — DIAGNOSIS lr sweep lr=1e-3 (TinyStories 12M))

_2026-11-13 15:25:27 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.001,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1240320 | 1209472 | 4575488 | 12001280 | 0.9319 ± 0.0100 | 2.539 ± 0.025 | 804447 ± 3274 | 286 ± 0 | 3 |
| A0-asca | 1833349 | 1212677 | 5940992 | 12001280 | 1.0481 ± 0.0056 | 2.852 ± 0.016 | 231490 ± 2559 | 680 ± 9 | 3 |


## Training komparasi (nano, 1465 step — DIAGNOSIS lr sweep lr=3e-3 (TinyStories 12M))

_2026-11-13 15:29:06 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1240320 | 1209472 | 4575488 | 12001280 | 0.9013 ± 0.0131 | 2.463 ± 0.032 | 772067 ± 5714 | 286 ± 0 | 3 |
| A0-asca | 1833349 | 1212677 | 5870720 | 12001280 | 0.9246 ± 0.0081 | 2.521 ± 0.021 | 231094 ± 2576 | 718 ± 13 | 3 |


## Training komparasi (nano, 1465 step — DIAGNOSIS lr sweep lr=6e-3 (TinyStories 12M))

_2026-11-13 15:32:51 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.006,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1240320 | 1209472 | 4575488 | 12001280 | 0.8875 ± 0.0065 | 2.429 ± 0.016 | 756320 ± 3180 | 286 ± 0 | 3 |
| A0-asca | 1833349 | 1212677 | 5767424 | 12001280 | 0.8973 ± 0.0085 | 2.453 ± 0.021 | 225238 ± 2012 | 741 ± 5 | 3 |


## Training komparasi (nano, 1465 step — DIAGNOSIS kapasitas n_experts=4)

_2026-11-13 15:36:03 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 4,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| A0-asca | 1438085 | 1210629 | 5357824 | 12001280 | 0.9310 ± 0.0070 | 2.537 ± 0.018 | 220100 ± 2567 | 627 ± 2 | 3 |


## Training komparasi (nano, 1465 step — DIAGNOSIS kapasitas n_experts=16)

_2026-11-13 15:39:40 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.003,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 16,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| A0-asca | 2623877 | 1216773 | 6936064 | 12001280 | 0.9078 ± 0.0085 | 2.479 ± 0.021 | 195573 ± 2278 | 864 ± 8 | 3 |


## Training komparasi (nano, 1465 step — DIAGNOSIS lr=1e-2 (TinyStories 12M))

_2026-11-13 15:44:41 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.01,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 8,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "dense-matched",
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B0-dense-matched | 1240320 | 1209472 | 4575488 | 12001280 | 0.8967 ± 0.0178 | 2.452 ± 0.044 | 787240 ± 15218 | 286 ± 0 | 3 |
| A0-asca | 1833349 | 1212677 | 5780096 | 12001280 | 0.8977 ± 0.0112 | 2.454 ± 0.027 | 207656 ± 4370 | 757 ± 2 | 3 |


## Training komparasi (nano, 1465 step — DIAGNOSIS lr=6e-3 + 16 expert)

_2026-11-13 15:47:48 · seed 0 · device cuda · torch 2.11.0+cu128_

```json
{
  "preset": "nano",
  "corpus": "file",
  "tokens_in_corpus": 19014224,
  "vocab": 241,
  "steps": 1465,
  "batch": 8,
  "seq_len": 1024,
  "train_tokens": 12001280,
  "lr": 0.006,
  "warmup": 100,
  "seed": 0,
  "device": "cuda",
  "torch": "2.11.0+cu128",
  "asca_config": {
    "vocab_size": 241,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "head_dim": 32,
    "max_seq_len": 1024,
    "bias": false,
    "dropout": 0.0,
    "init_std": 0.02,
    "use_semantic_encoder": true,
    "sem_conv_kernel": 4,
    "sem_reduction": 4,
    "use_hybrid_attention": true,
    "local_window": 128,
    "linear_chunk": 64,
    "local_attn_impl": "sdpa",
    "rope_theta": 10000.0,
    "use_moe": true,
    "n_experts": 16,
    "top_k": 2,
    "n_shared_experts": 1,
    "expert_ff": 64,
    "shared_ff": 256,
    "moe_capacity_factor": 1.5,
    "moe_impl": "grouped",
    "router_threshold": 0.0,
    "aux_loss_coef": 0.01,
    "z_loss_coef": 0.001,
    "dense_ff": 344,
    "use_memory": true,
    "n_memory_slots": 32,
    "mem_segment_len": 256,
    "tie_embeddings": true
  },
  "models": [
    "asca"
  ],
  "compile": true
}
```

| model | params | active_params | flops_per_token | tokens | val_loss | ppl | tokens_per_sec | peak_vram_mb | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| A0-asca | 2623877 | 1216773 | 6925312 | 12001280 | 0.8813 ± 0.0099 | 2.414 ± 0.024 | 210608 ± 1722 | 967 ± 5 | 3 |

