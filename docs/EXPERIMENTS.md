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

## Status ringkas v0.2 (setelah optimasi kernel, nano, 1.23M token, 3 seed)

| Aspek | Sebelum | Sesudah | Kesimpulan |
|---|---|---|---|
| Throughput ASCA | 63.0k tok/s | **106.1k ± 1.5k tok/s** | **1.68×** lebih cepat (grouped GEMM + chunk adaptif + `torch.compile`) |
| Throughput dense-matched | 344k tok/s | 534k ± 24k tok/s | 1.55× (compile juga membantu baseline — jarak tetap ~5×) |
| val loss | 0.4144 ± 0.0034 vs 0.4093 ± 0.0018 | **0.4165 ± 0.0020 vs 0.4167 ± 0.0063** | **setara** dengan dense param-matched (dalam noise) |
| FLOPs/token (param-matched) | 2.61M vs 2.97M | 3.65M vs 2.97M | +23% lebih boros; sebagian dari padding grouped GEMM |
| VRAM training (nano) | 175 MB vs 124 MB | 185 MB vs 99 MB | masih **1.9× lebih besar** |

Perubahan yang diukur (profil per-komponen, `benchmarks/profile_asca.py`, tiny T=512):

| Komponen | Sebelum | Sesudah | Perubahan |
|---|---|---|---|
| Attention (lokal + linear) | 8.24 ms | 4.09 ms | **2.0×** (chunk 64→128 efektif) |
| MoE | 3.78 ms | 2.22 ms | **1.7×** (grouped GEMM: 2 `bmm` total, bukan 3·E kernel) |
| Model penuh (eager) | 71.7 ms | 61.8 ms | 1.16× |
| Model penuh (+`torch.compile`) | — | 42.9 ms | **1.54×** tambahan |

**Yang masih menjadi masalah (jujur):**

1. **VRAM belum turun** — mask window eksplisit T×T dan skor memori O(S²LM) masih material,
   dan `torch.compile` menambah buffer kerja. Jalur FlexAttention sudah tersedia
   (`local_attn_impl="flex"`) tapi baru menang saat dikompilasi dan T besar.
2. **Gap wall-clock vs dense masih ~2.6–5×** pada d_model kecil: ASCA menjalankan 4 sublayer
   per blok dengan matmul kecil, sementara baseline adalah GEMM besar. Ini masalah
   *kernel/occupancy*, bukan masalah FLOPs (FLOPs/token hanya +7% pada active params setara
   di preset tiny).
3. **Grouped GEMM membayar padding** yang bergantung keseimbangan routing:
   terukur 1.07× (init seimbang) sampai ~1.65× (routing miring). `moe_pad_factor`
   dilaporkan di `bench_flops.py`; `moe_impl="loop"` tersedia sebagai jalur yang
   FLOPs-nya persis sebanding token.
4. **Belum ada KV-cache**: decode masih recompute penuh (10.4 ms/tok).

**Belum ada klaim keunggulan.** Yang sudah terbukti: mekanisme ASCA benar secara
matematis (kausal + cocok dengan referensi naif) dan kualitasnya setara baseline pada
budget ini. Yang belum: efisiensi wall-clock, VRAM, dan latensi decode.



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

