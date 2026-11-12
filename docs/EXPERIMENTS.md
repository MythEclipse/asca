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

## Status ringkas v0.1 (run pertama, nano, 1.23M token, 3 seed)

| Aspek | Hasil | Kesimpulan |
|---|---|---|
| Kualitas (val loss) | ASCA `0.4144 ± 0.0034` vs dense-matched `0.4093 ± 0.0018` | Setara dalam noise; dense sedikit unggul. **Belum ada bukti keunggulan.** |
| Throughput training | ASCA ~63k tok/s vs dense ~344k tok/s | ASCA **5.5× lebih lambat** (overhead kernel: loop chunk, gather/scatter MoE, einsum memori). |
| VRAM training T=4096 | ASCA 783 MB vs dense-matched 456 MB | **Belum ada keunggulan memori**: mask window eksplisit T×T dan skor memori O(S²LM) masih material. |
| Latensi decode | ASCA 10.4 ms/tok vs dense 1.4 ms/tok | ASCA belum punya KV-cache → recompute penuh. Ini batas atas, bukan hasil akhir. |

**Interpretasi:** mekanisme ASCA sudah benar secara matematis (lulus uji kausalitas
dan uji kesesuaian dengan referensi naif), tetapi **implementasi belum efisien**.
Klaim efisiensi belum bisa dibuat dan tidak akan dibuat sampai:

1. Attention lokal memakai FlexAttention/FlashAttention dengan window native
   (menghapus mask T×T dan alokasi skor penuh).
2. Attention linear tidak lagi loop per chunk (fused kernel / `torch.compile`).
3. MoE memakai dispatch ter-fusi, bukan loop per expert.
4. Decode memakai state rekuren (bukan recompute penuh) + KV-cache untuk jalur lokal.


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

