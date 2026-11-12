#!/usr/bin/env python3
"""Benchmark FLOPs & throughput: ASCA vs Dense Transformer.

Mengukur, bukan mengestimasi:
  * total & active parameter
  * FLOPs forward per token (FlopCounterMode — menghitung matmul yang benar-benar jalan)
  * throughput training (tokens/sec, fwd+bwd, bf16)
  * peak VRAM training

Contoh:
    env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_flops.py --preset tiny
    ... --ablation A3        # bandingkan ASCA tanpa MoE
    ... --out docs/EXPERIMENTS.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    append_experiment,
    build_asca,
    build_dense,
    count_flops_per_token,
    count_params,
    dense_matched_to,
    get_device,
    md_table,
    set_seed,
)


def measure_throughput(model, vocab, seq_len, batch, device, iters=8, warmup=3) -> float:
    """tokens/sec untuk satu langkah training (fwd+bwd, bf16)."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    y = torch.randint(0, vocab, (batch, seq_len), device=device)

    def step():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
            aux = getattr(model, "aux_loss", None)
            total = loss if aux is None else loss + aux
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / iters
    return batch * seq_len / secs


def describe(model, vocab, seq_len, batch, device, label, iters=8) -> dict:
    total, per_tok = count_flops_per_token(model, seq_len, vocab)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tps = measure_throughput(model, vocab, seq_len, batch, device, iters=iters)
    peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    row = {
        "model": label,
        "params": count_params(model),
        "active_params": model.active_params_per_token(),
        "flops_per_token": per_tok,
        "train_tokens_per_sec": round(tps, 1),
        "peak_vram_mb": round(peak, 1),
    }
    print(
        f"  {label:28s} params={row['params']:>10,} active={row['active_params']:>10,} "
        f"flops/tok={per_tok:>12,} tok/s={tps:>9.1f} vram={peak:>7.1f}MB",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=["nano", "tiny", "small"])
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--ablation", default=None, help="A1..A6 (default: ASCA penuh)")
    ap.add_argument("--out", default=None, help="tulis hasil ke docs/EXPERIMENTS.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    seq_len = args.seq_len or {"nano": 256, "tiny": 512, "small": 1024}[args.preset]
    print(f"device={device} preset={args.preset} seq_len={seq_len} batch={args.batch}\n")

    from asca import ASCAModel, ablation as make_ablation

    from common import asca_config

    cfg = asca_config(args.preset, args.vocab, seq_len)
    label_asca = "ASCA-full"
    if args.ablation:
        cfg = make_ablation(cfg, args.ablation)
        label_asca = f"ASCA-{args.ablation}"
    asca = ASCAModel(cfg).to(device)
    dense = build_dense(args.preset, args.vocab, seq_len).to(device)
    matched = dense_matched_to(asca, args.preset, seq_len).to(device)

    rows = [
        describe(dense, args.vocab, seq_len, args.batch, device, "B0-dense (default)"),
        describe(matched, args.vocab, seq_len, args.batch, device, "B0-dense (param-matched)"),
        describe(asca, args.vocab, seq_len, args.batch, device, label_asca),
    ]

    print("\n" + md_table(rows))
    if args.out:
        append_experiment(
            args.out,
            f"FLOPs & throughput ({args.preset}, seq {seq_len})",
            rows,
            {
                "preset": args.preset,
                "seq_len": seq_len,
                "batch": args.batch,
                "vocab": args.vocab,
                "seed": args.seed,
                "device": str(device),
                "ablation": args.ablation,
                "asca_config": asca.cfg.to_dict(),
            },
        )
        print(f"Hasil ditambahkan ke {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
