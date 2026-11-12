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

