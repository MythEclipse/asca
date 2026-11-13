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

## Not done yet

- **No dense baseline in this crate yet.** Without it there is no legitimate
  efficiency comparison — the same mistake v1 made. This is the next step.
- No CUDA kernels; matmuls are naive scalar loops.
- No loss-vs-tokens curve against a baseline.
- `w_gate` / `b_gate` receive very small gradients (scale 4.6e-4 vs 1.3e0 for
  `b_qkv`); whether the gates actually learn needs checking.
