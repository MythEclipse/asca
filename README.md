# ASCA — Adaptive Sparse Cognitive Architecture

From-scratch LLM architecture research (*random init*, no pretrained weights) targeting
compute and memory efficiency. **Everything is Rust, with zero external dependencies**:
training, checkpoint loading, inference and the CUDA kernels. There is no Python left in
this repository — see [Archive](#archive-v1-pytorch) for why it was removed.

## Status: v2 "Delta Memory Core"

No attention, no softmax in the feature path, no tokenizer, constant-size decode state.

| Aspect | v1 (removed) | v2 (this repo) |
|---|---|---|
| Attention | hybrid softmax + linear | **none** |
| Decode state | KV cache, O(T) | **S of size d×d per head, O(1)** |
| Tokenizer | byte via library | **byte-level, none** |
| MoE routing | learned top-k + aux loss | **hash(id) % n_experts**, zero FLOP |
| Training | PyTorch autograd | **hand-written BPTT backward** |
| Dependencies | torch, numpy, tokenizers | **zero** |

Core recurrence (gated delta rule — an associative memory that *overwrites* slots instead
of accumulating blindly):

```text
u_t = beta_t * (v_t - alpha_t * S_{t-1}^T k_t)
S_t = alpha_t * S_{t-1} + k_t u_t^T
o_t = S_t^T q_t
```

A block is `x = x + Delta(RMSNorm(x))`, then `x = x + HashMoE(RMSNorm(x))`. See
[`docs/SPEC_V2.md`](docs/SPEC_V2.md) and [`rust/README.md`](rust/README.md).

### Correctness gates

- Recurrent and chunked forms of the delta rule agree to **1.8e-15** (fp64, Python oracle
  at the time; the oracle is gone, the gate lives on as `tests/gradcheck*.rs`).
- All hand-written gradients match finite differences on the largest-gradient entries
  across every parameter: `gradcheck` (delta block), `gradcheck_model` (full model),
  `gradcheck_dense` (baseline).

### Quality: same-engine comparison, and the honest curve

Both models live in this crate with the same matmul, optimizer, data, seeds and compiler
flags. At 205k tokens ASCA looked clearly ahead (Δ −0.24 at matched total params). A
longer run shows most of that was an early-training effect:

| tokens | ASCA val loss | dense val loss | Δ | t (df=2) |
|---|---|---|---|---|
| 128k | 1.845 | 2.078 | **−0.233 ± 0.031** | −13.0 |
| 384k | 1.591 | 1.694 | −0.104 ± 0.078 | −2.3 |
| 512k | 1.499 | 1.574 | −0.075 ± 0.046 | −2.8 |
| 1.024M | **1.297 ± 0.087** | **1.369 ± 0.060** | **−0.072 ± 0.032** | −4.0 |

The gap **shrinks 69%** and flattens around −0.05…−0.08 with **no crossover up to 1M
tokens** — but t = −4.0 with 3 seeds is p ≈ 0.058, so **no quality advantage can be
claimed yet**. The earlier "ASCA wins by 0.24" is retracted.

What *is* measured and stable: at matched total parameters ASCA trains **1.66× faster**
(6,580 vs 3,973 tok/s) with a decode state of **2,048 floats that does not grow with
context**, versus 32,768 for the dense KV cache.

A bug in the baseline nearly made the first number look better than it was: `rope_apply`
rotated only the first head's slice, so 3 of 4 heads had no positional information.
Fixing it shrank the gap from −0.35 to −0.24. **When the baseline loses, audit the
baseline first.** Full detail in [`docs/RESULTS_V2.md`](docs/RESULTS_V2.md).

### CUDA: kernels verified, no end-to-end speedup yet

Tiled GEMM, transposed-A GEMM, RMSNorm fwd/bwd, elementwise, and the delta-rule scan as
one block per head with the whole time loop inside the kernel. Parity with CPU is
≤ 2.6e-7 relative and training loss is identical; GEMM is **22× faster** at
4096×1024×1024. It does **not** speed up training yet — 6,856 tok/s CPU vs 6,829 tok/s
GPU — because a step is 74.7 ms while its matmuls are only ~13 ms (~18%); the rest is
elementwise work, the delta scan, and one H2D/D2H copy plus sync per op. See
[`docs/CUDA.md`](docs/CUDA.md) for measurements and the fix list.

## Repository layout

```
asca/
├── Makefile                 # build / test / gpu / train / bench / curve
├── scripts/fetch_corpus.sh  # download TinyStories, print the path
├── docs/
│   ├── SPEC_V2.md           # v2 architecture and rationale
│   ├── RESULTS_V2.md        # protocol, numbers, caveats
│   ├── CUDA.md              # kernel parity, GEMM benchmark, cost breakdown
│   ├── EXPERIMENTS.md       # v1 experiment log (archive)
│   └── SPEC_ASCA.md         # v1 specification (archive)
└── rust/
    ├── build.rs             # nvcc -ptx, embeds PTX; crate builds without it
    ├── kernels/asca.cu      # 8 kernels
    ├── src/
    │   ├── tensor.rs        # Mat, Param, Rng, AdamW, RMSNorm
    │   ├── delta.rs         # DeltaBlock: forward + hand-written BPTT
    │   ├── model.rs         # hash-routed MoE, residual stack, byte head
    │   ├── dense.rs         # causal Transformer baseline, same engine
    │   ├── bench.rs         # Trainable trait + one loop for both architectures
    │   ├── train.rs         # data, training loop, checkpoint save/load
    │   ├── cuda/            # driver API bindings + Gpu dispatch
    │   └── main.rs          # train / eval / sample / inspect / bench / gpu
    └── tests/               # gradcheck, gradcheck_model, gradcheck_dense
```

## Running it

```bash
make corpus                  # TinyStories -> ~/.cache/asca/TinyStories-valid.txt
make build                   # cargo build --release
make test                    # 3 finite-difference gradient checks
make gpu                     # CUDA parity + GEMM benchmark
make bench STEPS=400         # ASCA vs dense, same engine, 3 seeds
make curve STEPS=2000        # loss-vs-tokens curve
```

Direct binary:

```bash
rust/target/release/asca train --data "$(./scripts/fetch_corpus.sh)" \
    --steps 1200 --seq-len 128 --batch 4 --lr 6e-3 --out /tmp/asca.ckpt
ASCA_GPU=0 rust/target/release/asca bench ...   # force the CPU path
```

## Evaluation protocol

- **Language**: validation loss & perplexity on the same held-out corpus.
- **Efficiency**: training tokens/sec, GEMM GFLOPS, decode state size in floats.
- **Statistics**: ≥3 seeds as mean ± std, paired per seed, with a t statistic. A single
  run is not a claim. Report the noise floor.
- **LR must be tuned per architecture** before any comparison — one shared LR produced a
  false "ASCA loses" result in v1.
- **Synthetic-corpus ablations are not used for architecture decisions.** Real text only.
- **No efficiency claim without a baseline in the same engine.**

## Roadmap

1. **Make the GPU path actually pay off**: device-resident tensors (one upload/download
   per step), fused elementwise, cross-sequence batching (512 rows → GEMM 4× bigger),
   GPU delta backward. Estimated 3–6× end-to-end.
2. **Long curve**: 2–4M tokens with 5+ seeds — does the residual −0.07 survive, shrink to
   zero, or invert?
3. **Long-range recall test** (copy / associative recall), where a compressed state should
   lose to attention. This is the honest stress test of the whole thesis.
4. **Scale**: 331k params at 1M tokens is a toy regime. Nothing about architecture should
   be claimed before ≥100M tokens.

---

## Archive: v1 (PyTorch)

v1 was a Transformer variant with hybrid softmax + linear attention, a learned MoE router
with aux loss, and a tokenizer. **Its code was deleted from this repository** in the Rust
migration; the specification and the experiment log remain in
[`docs/SPEC_ASCA.md`](docs/SPEC_ASCA.md) and
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) as a historical record.

Why it was removed:

- **The efficiency numbers measured PyTorch, not the architecture.** ~2,900 kernel
  launches per decode token, plus dispatcher overhead and tiny GEMMs. Any efficiency
  comparison inside that framework is a measurement of the framework.
- **It was not different enough**: still attention, still softmax, still a tokenizer,
  still a learned router.

What v1 measured before it was retired (TinyStories, 12–24M tokens, 3–5 seeds):

| lr (shared) | dense-matched | ASCA (8 experts) | Δ |
|---|---|---|---|
| 1e-3 | 0.9319 | 1.0481 | +0.1162 |
| 3e-3 | 0.9013 | 0.9246 | +0.0234 |
| 6e-3 | **0.8875** | **0.8973** | **+0.0098** |
| 1e-2 | 0.8967 | 0.8977 | **+0.0011** |

ASCA with 16 experts at lr 6e-3 reached **0.8813** — nominally the best number in the
table, inside the dense noise band. **Conclusion: no quality difference was detectable in
either direction.** A v0.7 claim that ASCA was significantly worse (p = 0.042) was
retracted: it came from forcing one shared learning rate on both architectures.

Cost at its best quality point: **3.7× slower to train and 3.4× more VRAM** than the dense
control. For an architecture whose thesis is efficiency, that was the real problem.

What still holds from v1: parallel prefill at **48,955 tok/s**, **1.60 ms/token** decode
with CUDA Graphs, and a **constant 3.8 MB decode cache vs 24 MB** for dense.
