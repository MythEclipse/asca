# ASCA — Adaptive Sparse Cognitive Architecture

Riset arsitektur LLM dari nol (*random init*, tanpa bobot pretrained) yang mengejar
efisiensi komputasi dan memori melalui mekanisme adaptif dan sparse.

> **Status: v0.7 — hasil NEGATIF pada teks nyata.**
>
> Pada TinyStories (byte-level, 24M token, seq 1024, **5 seed**):
>
> | | val loss | throughput | VRAM |
> |---|---|---|---|
> | dense-matched (kontrol) | **0.8065 ± 0.0105** | **769k tok/s** | **286 MB** |
> | ASCA penuh | 0.8282 ± 0.0069 | 224k tok/s | 709 MB |
>
> Paired t = 2.95, df = 4, **p = 0.042** — kelima seed lebih buruk. ASCA 3.4× lebih
> lambat dan 2.5× lebih boros VRAM. **Tidak ada klaim keunggulan.**
>
> **Pelajaran metodologis:** semua eksperimen awal memakai korpus sintetis, dan di
> sana ASCA tampak setara bahkan sedikit lebih baik (Δ −0.002…−0.009). Kesimpulan
> itu **tidak bertahan** pada teks nyata — arah tandanya berbalik, dan varian yang
> tampak "sama saja" di sintetis jadi +0.09 lebih buruk. **Ablasi sintetis tidak
> dipakai lagi untuk keputusan arsitektur.**
>
> Yang sudah terukur dan bertahan: **prefill paralel 48.955 tok/s** (1.32× lebih
> lambat dari dense) dan **decode 1.60 ms/token** dengan CUDA Graph, dengan **cache
> decode 3.8 MB konstan vs 24 MB** milik dense. Semua detail di
> [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## 1. Visi Utama

Foundation model general-purpose dengan prioritas:
- **Language & Semantic Primacy** — makna dan relasi bahasa sebagai fondasi.
- **Reasoning & Logic**, **Algorithms & Code**, **Mathematics & Science**.
- **Long-Context Efficiency** — skalabilitas konteks tanpa ledakan memori $O(N^2)$.
- **High-Throughput Inference** — latensi decode rendah pada GPU konsumen (target: 6 GB).

## 2. Prinsip Pengembangan

1. **Dari Nol**: semua bobot inisialisasi acak, tanpa pretrained checkpoint.
2. **Mekanisme Mandiri**: operator internal dirumuskan sendiri, bukan sekadar ubah config Transformer.
3. **Efisiensi sebagai Metrik Utama**: kualitas diukur relatif terhadap FLOPs, waktu, dan VRAM — bukan jumlah parameter.
4. **Validasi Berbasis Kontrol**: setiap klaim wajib head-to-head vs Dense Transformer dan MoE standar pada budget token/FLOPs identik.
5. **Kausalitas Non-Negotiable**: setiap mekanisme wajib lulus uji kebocoran token masa depan sebelum masuk training.

## 3. Komponen Inti

| Komponen | Mekanisme (lihat spesifikasi) |
|---|---|
| **Semantic Encoder** | Konvolusi kausal depthwise + bottleneck MLP sekali di input (§3.1) |
| **Adaptive Attention** | Sliding-window causal attention ⊕ gated linear attention, digabung gate data-dependent (§3.2) |
| **Adaptive Router** | Top-$k$ routing + threshold skip (adaptive compute) + switch/z-loss (§3.3) |
| **Sparse Experts** | Shared expert selalu aktif + routed SwiGLU experts, dispatch per-expert (§3.3) |
| **Efficient Memory** | SegmentMemory: ringkasan segmen dengan $M$ slot, dibaca kausal antar-segmen (§3.4) |
| **Shared Decoder** | RMSNorm + gated semantic skip + proyeksi tertaut embedding (§3.5) |

Spesifikasi matematis lengkap (notasi, dataflow tensor, anggaran FLOPs, matriks ablasi):
[`docs/SPEC_ASCA.md`](docs/SPEC_ASCA.md).

## 4. Struktur Repositori

```
asca/
├── docs/
│   ├── SPEC_ASCA.md            # Formulasi matematis + matriks ablasi
│   └── EXPERIMENTS.md          # Log hasil benchmark (append otomatis)
├── asca/                       # Implementasi arsitektur
│   ├── config.py               # ASCAConfig + definisi ablasi A0..A6
│   ├── norm.py                 # RMSNorm, RoPE
│   ├── encoder.py              # Semantic Encoder
│   ├── attention.py            # Adaptive Hybrid Attention (lokal + linear chunked)
│   ├── router.py               # Adaptive Router + aux loss
│   ├── experts.py              # Shared + routed sparse experts (MoE)
│   ├── memory.py               # SegmentMemory
│   ├── block.py                # Pre-norm residual block
│   └── model.py                # ASCAModel + SharedDecoder + generate
├── baseline/
│   └── dense_transformer.py    # Kontrol: Llama/GPT-style dense causal transformer
├── data/
│   ├── tokenizer.py            # BPE (tokenizers) + fallback byte-level
│   └── dataloader.py           # Korpus sintetis / file / HF + TokenDataset
├── benchmarks/
│   ├── common.py               # Preset, param matching, FLOP counter, loop training
│   ├── bench_flops.py          # FLOPs/token & throughput terukur
│   ├── bench_vram.py           # Peak VRAM 512..4096 & latensi decode
│   ├── profile_asca.py         # Profil per-komponen (cari hotspot)
│   ├── bench_decode.py         # Latensi decode + memori cache (recompute/eager/graph)
│   └── train_toy.py            # Training komparasi multi-seed (utama)
└── tests/
    └── run_tests.py            # Uji kausalitas, referensi naif, shapes, MoE, dll.
```

## 5. Cara Menjalankan

Environment (wajib memakai venv bersama, **bukan** `.venv` per proyek):

```bash
source ~/.venvs/ml-activate        # torch 2.11.0+cu128, RTX 4050 sm_89
```

Test (12 test, ~2 detik di GPU):

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python tests/run_tests.py
```

Benchmark FLOPs & throughput:

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_flops.py --preset tiny --out docs/EXPERIMENTS.md
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_flops.py --ablation A3     # ablasi
```

VRAM & latensi decode:

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_vram.py --preset nano --out docs/EXPERIMENTS.md
```

Profil hotspot per komponen:

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/profile_asca.py --preset tiny --seq-len 512
```

Training komparasi (utama) — `--compile` memberi ~1.5× tambahan:

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/train_toy.py \
    --preset nano --steps 600 --batch 8 --seq-len 256 \
    --models dense-matched,asca --seeds 0,1,2 --compile --out docs/EXPERIMENTS.md
```

Korpus nyata (perlu `datasets` di venv bersama untuk streaming HF):

```bash
... --corpus hf --hf-repo HuggingFaceFW/fineweb --max-chars 50000000
```

## 6. Protokol Evaluasi

- **Language**: validation loss & perplexity pada korpus uji yang sama.
- **Efficiency**: active FLOPs/token (terukur), tokens/sec training, ms/token decode.
- **Memory**: peak VRAM pada $T \in \{512, 1024, 2048, 4096\}$.
- **Reasoning/Code/Math**: GSM8K, HumanEval, MBPP — hanya setelah skala memadai.
- **Statistik**: minimal 3 seed, dilaporkan mean ± std. Satu run tidak diklaim.

## 7. Roadmap Terdekat

Sudah selesai (v0.2): grouped-GEMM MoE dispatch (1.7×), chunk adaptif (2.0× pada
attention), jalur FlexAttention opsional, `torch.compile` opt-in (1.54×).

Sudah selesai (v0.5): KV-cache + state rekuren, fix normalisasi `SegmentMemory`,
CUDA Graph decode, dan **prefill paralel** (state cache diturunkan dari lintasan
chunked; 611 → 48.955 tok/s).

1. **Diagnosis defisit**: apakah ASCA kalah karena kapasitas efektif lebih kecil
   (active params 0.9M tersebar di 4 sublayer vs satu MLP lebar), karena optimisasi
   (LR/warmup belum di-tuning per arsitektur), atau karena skalanya terlalu kecil?
   Kandidat eksperimen: samakan **total** params (bukan active), sweep LR, dan
   kurva loss vs token (apakah ASCA menyusul dengan training lebih lama).
2. **Skala**: 24M token & d=128 masih rezim mainan. Perlu ≥100M token dan preset
   `small` sebelum ada klaim apa pun.
3. **Pertahankan yang sudah menang**: cache decode konstan (3.8 MB vs 24 MB) dan
   prefill paralel — tapi itu belum cukup untuk membenarkan arsitekturnya.
4. **Turunkan VRAM training**: FlexAttention sebagai default pada T besar, ganti skor
   memori O(S²LM) dengan pembacaan per-segmen.
5. **Ablasi penuh A0–A6** pada ≥100M token (sekarang baru 300–600 step, 1 seed/3 seed).
6. **Skala**: preset `small` (seq 1024) lalu Colab (A100/T4) untuk pretraining
   dengan checkpoint ke Drive/HF Hub + resume otomatis.
