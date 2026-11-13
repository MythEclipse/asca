# ASCA — Adaptive Sparse Cognitive Architecture

From-scratch LLM architecture research (*random init*, no pretrained weights) targeting
compute and memory efficiency through adaptive and sparse mechanisms.

> **v2 is now a from-scratch Rust implementation.** The v1 code in this repo is
> PyTorch, and that was a methodological dead end: the efficiency numbers measured
> PyTorch's dispatcher (~2,900 kernel launches per decode token), not the architecture,
> and the model was still a Transformer variant with attention, softmax, a tokenizer
> and a learned router. So the architecture was redesigned and reimplemented with **no
> external dependencies**: training, checkpoint loading and inference are all Rust.
>
> **ASCA v2 (Delta Memory Core)** — no attention, no softmax in the feature path, no
> tokenizer, constant-size state:
>
> | Aspect | v1 (PyTorch) | v2 (pure Rust) |
> |---|---|---|
> | Attention | hybrid softmax + linear | **none** |
> | Decode state | KV cache, O(T) | **S of size d x d per head, O(1)** |
> | Tokenizer | byte via library | **byte-level, none** |
> | MoE routing | learned top-k + aux loss | **hash(id) % n_experts**, zero FLOP |
> | Training | PyTorch autograd | **hand-written BPTT backward** |
> | Dependencies | torch, numpy, tokenizers | **zero** |
>
> Core recurrence (gated delta rule — an associative memory that *overwrites* slots
> instead of accumulating blindly):
>
> ```text
> u_t = beta_t * (v_t - alpha_t * S_{t-1}^T k_t)
> S_t = alpha_t * S_{t-1} + k_t u_t^T
> o_t = S_t^T q_t
> ```
>
> Verified so far: recurrent and chunked forms agree to **1.8e-15**; all hand-written
> gradients match finite differences on 40 entries across every parameter; a 331k-param
> model trains on TinyStories bytes at **6,583 tok/s on CPU** (naive matmul, no GPU) and
> reaches **val loss 1.379 / ppl 3.97** in 614k tokens. See
> [`docs/SPEC_V2.md`](docs/SPEC_V2.md) and [`rust/`](rust/README.md).
>
> **Same-engine comparison, and the honest curve.** Both models live in this crate with
> the same matmul, optimizer, data, seeds and flags (TinyStories bytes, 3 seeds). At
> 205k tokens ASCA looked clearly ahead (Δ −0.24 at matched total params). A longer run
> shows most of that was an early-training effect:
>
> | tokens | ASCA val loss | dense val loss | Δ | t (df=2) |
> |---|---|---|---|---|
> | 128k | 1.845 | 2.078 | **−0.233 ± 0.031** | −13.0 |
> | 384k | 1.591 | 1.694 | −0.104 ± 0.078 | −2.3 |
> | 512k | 1.499 | 1.574 | −0.075 ± 0.046 | −2.8 |
> | 1.024M | **1.297 ± 0.087** | **1.369 ± 0.060** | **−0.072 ± 0.032** | −4.0 |
>
> The gap **shrinks 69%** and flattens around −0.05…−0.08 with **no crossover up to 1M
> tokens** — but t = −4.0 with 3 seeds is p ≈ 0.058, so **no quality advantage can be
> claimed yet**. The earlier "ASCA wins by 0.24" is retracted.
>
> What *is* measured and stable: at matched total parameters ASCA trains **1.66× faster**
> (6,580 vs 3,973 tok/s) with a decode state of **2,048 floats that does not grow with
> context**, versus 32,768 for the dense KV cache.
>
> A bug in the baseline nearly made the first number look better than it was:
> `rope_apply` rotated only the first head's slice, so 3 of 4 heads had no positional
> information. Fixing it shrank the gap from −0.35 to −0.24. **When the baseline loses,
> audit the baseline first.** Full detail in [`docs/RESULTS_V2.md`](docs/RESULTS_V2.md).
>
> **CUDA kernels are written and verified** (tiled GEMM, transposed-A GEMM, RMSNorm fwd/bwd,
> elementwise, and the delta-rule scan as one block per head with the whole time loop
> inside the kernel). Parity with CPU is ≤ 2.6e-7 relative and training loss is identical;
> GEMM is **20.5× faster** at 4096×1024×1024. It does **not** speed up training yet —
> 6,856 tok/s CPU vs 6,829 tok/s GPU — because a step is 74.7 ms while its matmuls are
> only ~13 ms (~18%); the rest is elementwise work, the delta scan, and one H2D/D2H copy
> plus sync per op. See [`docs/CUDA.md`](docs/CUDA.md) for the measurements and the fix
> list.
>
> Still open: 2–4M token runs with 5+ seeds (does the residual −0.07 survive?) and a
> long-range recall test (where attention should win). Absolute tok/s here is scalar CPU
> code and means nothing outside this crate. The v1 research record
> follows below.
>
> **Status: v0.8 — the earlier negative result is retracted; it was mostly a learning-rate artifact.**
>
> v0.7 claimed ASCA was significantly worse than the dense control (p = 0.042). That
> was measured with **one shared learning rate (3e-3) for both architectures**. Tuning
> the LR per architecture on the same corpus (TinyStories, 12M tokens, 3 seeds) gives:
>
> | lr | dense-matched | ASCA (8 experts) | Δ |
> |---|---|---|---|
> | 1e-3 | 0.9319 | 1.0481 | +0.1162 |
> | 3e-3 | 0.9013 | 0.9246 | +0.0234 |
> | 6e-3 | **0.8875** | **0.8973** | **+0.0098** |
> | 1e-2 | 0.8967 | 0.8977 | **+0.0011** |
>
> ASCA with 16 experts at lr 6e-3 reaches **0.8813** — nominally the best number in the
> table, inside the noise band of the dense control.
>
> **Current claim status: no quality difference is detectable in either direction** —
> not "ASCA is worse", not "ASCA is better". What is measured and unambiguous is the
> **cost**: at its best quality point ASCA is **3.7× slower to train and uses 3.4× more
> VRAM** than the dense control. For an architecture whose thesis is efficiency, that
> is the real problem.
>
> Two protocol rules came out of this and are now binding: **LR must be tuned per
> architecture before any claim**, and **ablations on the synthetic corpus are not used
> for architecture decisions** (see v0.7 below in
> [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)).
>
> What still holds from earlier work: **parallel prefill at 48,955 tok/s** (1.32× slower
> than dense) and **1.60 ms/token decode** with CUDA Graphs, with a **constant 3.8 MB
> decode cache vs 24 MB** for dense.

---

## 1. Vision

A general-purpose foundation model with these priorities:

- **Language & Semantic Primacy** — meaning and linguistic relations as the foundation.
- **Reasoning & Logic**, **Algorithms & Code**, **Mathematics & Science**.
- **Long-Context Efficiency** — context scaling without the $O(N^2)$ memory blow-up.
- **High-Throughput Inference** — low decode latency on consumer GPUs (target: 6 GB).

## 2. Development Principles

1. **From Scratch**: all weights randomly initialized, no pretrained checkpoint.
2. **Self-Contained Mechanisms**: internal operators are formulated from first
   principles, not just Transformer config tweaks.
3. **Efficiency as the Primary Metric**: quality is measured relative to FLOPs, wall
   time, and VRAM — not parameter count.
4. **Control-Based Validation**: every claim must be a head-to-head comparison against
   a dense Transformer and a standard MoE at identical token/FLOP budgets.
5. **Non-Negotiable Causality**: every mechanism must pass a future-token-leakage test
   before it is allowed into training.

## 3. Core Components

| Component | Mechanism (see the spec) |
|---|---|
| **Semantic Encoder** | Depthwise causal convolution + bottleneck MLP, applied once at the input (§3.1) |
| **Adaptive Attention** | Sliding-window causal attention ⊕ gated linear attention, combined by a data-dependent gate (§3.2) |
| **Adaptive Router** | Top-$k$ routing + threshold skip (adaptive compute) + switch/z-loss (§3.3) |
| **Sparse Experts** | Always-on shared expert + routed SwiGLU experts, per-expert dispatch (§3.3) |
| **Efficient Memory** | SegmentMemory: per-segment summaries with $M$ slots, read causally across segments (§3.4) |
| **Shared Decoder** | RMSNorm + gated semantic skip + embedding-tied projection (§3.5) |

Full mathematical specification (notation, tensor dataflow, FLOP budget, ablation
matrix): [`docs/SPEC_ASCA.md`](docs/SPEC_ASCA.md).

## 4. Repository Layout

```
asca/
├── docs/
│   ├── SPEC_ASCA.md            # Mathematical formulation + ablation matrix
│   └── EXPERIMENTS.md          # Benchmark log (auto-appended)
├── asca/                       # Architecture implementation
│   ├── config.py               # ASCAConfig + ablation definitions A0..A9
│   ├── norm.py                 # RMSNorm, RoPE
│   ├── encoder.py              # Semantic Encoder
│   ├── attention.py            # Adaptive Hybrid Attention (local + chunked linear)
│   ├── router.py               # Adaptive Router + aux loss
│   ├── experts.py              # Shared + routed sparse experts (MoE)
│   ├── memory.py               # SegmentMemory
│   ├── block.py                # Pre-norm residual block
│   ├── graph_decode.py         # CUDA-graph decode wrapper
│   └── model.py                # ASCAModel + SharedDecoder + prefill/generate
├── baseline/
│   └── dense_transformer.py    # Control: Llama/GPT-style dense causal transformer
├── data/
│   ├── tokenizer.py            # BPE (tokenizers) + byte-level fallback
│   └── dataloader.py           # Synthetic / file / HF corpus + TokenDataset
├── benchmarks/
│   ├── common.py               # Presets, param matching, FLOP counter, training loop
│   ├── bench_flops.py          # Measured FLOPs/token & throughput
│   ├── bench_vram.py           # Peak VRAM 512..4096 & decode latency
│   ├── profile_asca.py         # Per-component profiling (hotspot hunting)
│   ├── bench_decode.py         # Decode latency + cache memory (recompute/eager/graph)
│   └── train_toy.py            # Multi-seed comparative training (primary entry point)
└── tests/
    └── run_tests.py            # Causality, naive-reference, shape, MoE, decode parity
```

## 5. Running It

Environment (must use the shared venv, **not** a project-local `.venv`):

```bash
source ~/.venvs/ml-activate        # torch 2.11.0+cu128, RTX 4050 sm_89
```

Tests (20 tests, ~6 s on GPU):

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python tests/run_tests.py
```

FLOPs & throughput benchmark:

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_flops.py --preset tiny --out docs/EXPERIMENTS.md
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_flops.py --ablation A3     # ablation
```

VRAM & decode latency:

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_vram.py --preset nano --out docs/EXPERIMENTS.md
```

Decode benchmark (recompute vs eager cache vs CUDA graph, plus parallel prefill):

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_decode.py --preset tiny --contexts 512,1024,2048 --new-tokens 12
```

Per-component hotspot profiling:

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/profile_asca.py --preset tiny --seq-len 512
```

Comparative training (primary) — `--compile` adds ~1.5×:

```bash
env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/train_toy.py \
    --preset nano --steps 600 --batch 8 --seq-len 256 \
    --models dense-matched,asca --seeds 0,1,2 --compile --out docs/EXPERIMENTS.md
```

Real corpus (a local file works without extra dependencies; `--corpus hf` needs
`datasets` in the shared venv):

```bash
... --corpus file --path /path/to/TinyStories-valid.txt --max-chars 19000000
... --corpus hf --hf-repo HuggingFaceFW/fineweb --max-chars 50000000
```

## 6. Evaluation Protocol

- **Language**: validation loss & perplexity on the same held-out corpus.
- **Efficiency**: active FLOPs/token (measured), training tokens/sec, decode ms/token.
- **Memory**: peak VRAM at $T \in \{512, 1024, 2048, 4096\}$.
- **Reasoning/Code/Math**: GSM8K, HumanEval, MBPP — only once the scale justifies it.
- **Statistics**: at least 3 seeds, reported as mean ± std. A single run is not a
  claim. The noise floor must be measured, not assumed, and it depends on the corpus
  (~±0.0006 seed std on synthetic data vs ~±0.015 on real text).

## 7. Roadmap

Done (v0.2): grouped-GEMM MoE dispatch (1.7×), adaptive chunking (2.0× on attention),
optional FlexAttention path, opt-in `torch.compile` (1.54×).

Done (v0.5): KV-cache + recurrent decode state, `SegmentMemory` normalization fix,
CUDA-graph decode, and **parallel prefill** (decode cache derived from the chunked
forward pass; 611 → 48,955 tok/s).

Open, in priority order:

1. ~~Diagnose the deficit~~ **done (v0.8)**: it was mostly LR. Capacity also matters
   (4→8→16 experts improves monotonically). Next: 5+ seeds at the best LR per model to
   see whether the two are separable at all, and a loss-vs-tokens curve.
2. **Scale**: 24M tokens at d=128 is a toy regime. ≥100M tokens and the `small`
   preset are required before any claim.
3. **Keep what already wins**: constant decode cache (3.8 MB vs 24 MB) and parallel
   prefill — but that is not yet enough to justify the architecture.
4. **Cut training VRAM**: make FlexAttention the default at large $T$, and replace the
   $O(S^2LM)$ memory score with per-segment reads.
5. **Full A0–A9 ablation** at ≥100M tokens on a real corpus (current runs are 12–24M
   tokens, nano preset only).
6. **Scale up**: `small` preset (seq 1024), then Colab (A100/T4) for pretraining with
   checkpointing to Drive/HF Hub and automatic resume.
