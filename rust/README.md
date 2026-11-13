# asca-rs — ASCA v2 in pure Rust

From-scratch implementation of the ASCA v2 architecture: training, checkpoint
loading, evaluation and sampling. **No external dependencies** — `std` only.

## Architecture in one paragraph

No attention. No softmax in the feature path. No tokenizer. Each head keeps an
associative memory `S ∈ R^{d_h × d_h}` updated by the **gated delta rule**:
`u_t = beta_t (v_t - alpha_t S^T k_t)`, `S_t = alpha_t S_{t-1} + k_t u_t^T`,
`o_t = S_t^T q_t`. The `(v - S^T k)` correction makes the state overwrite existing
slots instead of accumulating blindly. Decode memory is constant in context length.
The FFN is a **hash-routed MoE** (`hash(token_id) % n_experts`): zero routing FLOPs,
perfectly balanced, no auxiliary loss. Input is raw bytes (vocab 256).

## Build and run

```bash
cargo build --release

# train (writes a checkpoint)
./target/release/asca train --data /path/to/text.txt \
    --steps 1200 --seq-len 128 --batch 4 --lr 3e-3 \
    --d-model 64 --n-heads 4 --n-layers 2 --n-experts 8 --d-ff 128 \
    --out asca2.ckpt

# evaluate a checkpoint
./target/release/asca eval --ckpt asca2.ckpt --data /path/to/text.txt --batches 20

# sample bytes
./target/release/asca sample --ckpt asca2.ckpt --prompt "Once upon a time" --tokens 200

# inspect config and parameter shapes
./target/release/asca inspect --ckpt asca2.ckpt
```

## Layout

| File | Contents |
|---|---|
| `src/tensor.rs` | row-major `Mat`, xorshift128+ RNG, RMSNorm forward/backward, AdamW (decoupled weight decay) |
| `src/delta.rs` | delta block, forward + hand-derived BPTT backward |
| `src/model.rs` | embedding → N × [delta block, hash-MoE] → RMSNorm → head, cross-entropy + backward |
| `src/train.rs` | byte-level data, training loop, eval, binary checkpoint save/load |
| `src/main.rs` | CLI |
| `tests/gradcheck.rs` | finite-difference check of the delta block |
| `tests/gradcheck_model.rs` | finite-difference check of the whole model |

## Tests

```bash
cargo test --release            # both gradient checks
```

Gradients are hand-written, so they are verified against central finite differences
on the **largest-gradient entries** of every parameter, with a noise-aware criterion:
truncation error scales like `eps^2` and shrinks with `eps`, while f32 noise scales
like `ulp(loss) / (2 eps)` and grows as `eps` shrinks, so `eps = 3e-4` sits at the
balance point. Entries below three times the noise floor are skipped rather than
failed.

## Measured (CPU, no GPU, naive matmul)

TinyStories bytes, `d_model=64 n_heads=4 n_layers=2 n_experts=8 d_ff=128`:

| Metric | Value |
|---|---|
| Parameters | 331,408 |
| Training throughput | 6,583 tok/s |
| Tokens trained | 614k |
| val_loss | 2.96 → 1.379 (ppl 3.97) |
| Decode state | 2 × 4 × 16² = 2,048 floats, constant in context |

## Baseline comparison (`asca bench`)

`src/dense.rs` is a causal Transformer baseline living in the same crate: same `Mat`,
`Param`, `Rng`, RMSNorm, AdamW, data and training loop, with RoPE and softmax
attention. Its gradients are verified by finite differences too
(`tests/gradcheck_dense.rs`).

```bash
./target/release/asca bench --data /path/to/text.txt --steps 400 --batch 4 \
    --seq-len 128 --lr 6e-3 --seeds 0,1,2
```

Result (TinyStories bytes, 3 seeds). Short run first, then the long one that corrects it:

| tokens | matching | ASCA val loss | dense val loss | Δ | ASCA tok/s | dense tok/s |
|---|---|---|---|---|---|---|
| 205k | total (331k) | 1.735 | 1.976 | −0.242 | 6,810 | 4,243 |
| 205k | active (99k) | 1.735 | 1.978 | −0.244 | 7,019 | 14,529 |
| 1.02M | total (331k) | 1.297 | 1.369 | **−0.072** | 6,580 | 3,973 |

The advantage **shrinks 69%** between 128k and 1M tokens and is **not significant** at
1M (t = −4.0, df 2, p ≈ 0.058). What holds: at matched total params ASCA trains 1.66×
faster and its decode state is 2,048 floats constant, versus 32,768 growing with
context.

Use `--eval-curve N` to record the loss curve every N steps and `--match total|active|both`
to pick the parameter matching.

See `../docs/RESULTS_V2.md` for the full protocol and caveats.

## Not done yet

- No loss-vs-tokens curve out to millions of tokens; dense may catch up.
- No long-range recall test (where attention usually beats a compressed state).
- No CUDA kernels; matmuls are naive scalar loops, so absolute tok/s means nothing
  outside this crate.
- Only 3 seeds; a strong claim needs 5+ and a measured noise floor.
- `w_gate` / `b_gate` receive very small gradients (scale 4.6e-4 vs 1.3e0 for
  `b_qkv`); whether the gates actually learn needs checking.
