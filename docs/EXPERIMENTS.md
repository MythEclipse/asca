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

## Status ringkas v0.4 (CUDA Graph decode, tiny)

| Aspek | ASCA | dense-matched | Kesimpulan |
|---|---|---|---|
| Decode eager (ms/tok, konteks 2036) | 8.18 | 2.05 | ASCA 4.0× lebih lambat |
| **Decode CUDA Graph** | **1.64 (datar)** | **0.64 (datar)** | ASCA **2.6×** lebih lambat, gap menyempit dari 4.0× |
| Speedup graph vs eager | **5.0×** | 3.2× | launch overhead memang bottleneck-nya |
| Prefill lewat cache (graph) | **611 tok/s** | 1543 tok/s | dari 103 tok/s (6×) |
| Memori cache decode | **3.82 MB konstan** | 24.25 MB (prealloc `max_seq_len`) | ASCA **6.3×** lebih kecil, O(1) terhadap konteks |
| val loss (nano, 3 seed) | 0.4159 ± 0.0020 | 0.4210 ± 0.0145 | setara (tidak berubah oleh optimasi decode) |
| Throughput training (nano) | 103.9k tok/s | 557k tok/s | tidak berubah: jalur training tak disentuh |

**Cara mencapainya (dan mengapa `torch.compile` tidak dipakai):** jalur `step` dibuat
*bebas cabang* — posisi/window memakai `scatter_`/`index_select` dengan indeks dari
tensor, ringkasan segmen memakai online softmax, MoE memakai dispatch terpilih
(`index_select`, tanpa `nonzero`/`.item()`), dan **semua update state in-place**.
Dengan itu satu CUDA graph bisa dipakai untuk semua posisi. `torch.compile` di jalur
yang sama justru 40× lebih lambat (guards + recompiles).

**Jebakan yang ditemukan:** kalau sebuah buffer state di-*rebind* ke tensor baru saat
capture, memori lama dilepas di tengah capture dan graph bisa menulis ke blok yang
masih dipakai tensor lain → data rusak dan `device-side assert` di `index_select`.
Gejalanya menyesatkan (muncul saat replay, bukan saat capture).

**Yang masih gagal (jujur):**

1. **Latensi decode masih 2.6× lebih lambat** dari dense (1.64 vs 0.64 ms/tok).
   GPU-side work ASCA memang lebih banyak (4 sublayer/blok + state O(Dh²) per head).
2. **Prefill masih token-demi-token** (611 tok/s vs dense 1543). Belum memakai
   forward chunked paralel untuk prompt.
3. **Throughput training & VRAM training tidak berubah** (5.4× lebih lambat, 1.9× VRAM).
4. **Belum ada benchmark skala besar** (≥100M token). **Tidak ada klaim keunggulan.**

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

