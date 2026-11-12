#!/usr/bin/env python3
"""Profiling per-komponen: di mana waktu ASCA sebenarnya habis.

Dipakai untuk memutuskan optimasi mana yang layak dikerjakan (bukan menebak).

Contoh:
    env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/profile_asca.py --preset tiny --seq-len 512
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import asca_config, build_asca, get_device, md_table, set_seed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from asca.norm import rope_cache  # noqa: E402


def _ms(fn: Callable[[], torch.Tensor], params: list, iters: int, warmup: int, backward: bool) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    def one() -> torch.Tensor:
        with amp:
            return fn()

    for _ in range(warmup):
        out = one()
        if backward:
            torch.autograd.grad(out.float().pow(2).mean(), params, allow_unused=True)
        del out
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = one()
        if backward:
            torch.autograd.grad(out.float().pow(2).mean(), params, allow_unused=True)
        del out
    end.record()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench(name: str, fn: Callable[[], torch.Tensor], params: list, iters: int = 10) -> dict:
    fwd = _ms(fn, params, iters, 3, backward=False)
    both = _ms(fn, params, iters, 3, backward=True)
    return {"komponen": name, "fwd_ms": round(fwd, 3), "fwd+bwd_ms": round(both, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=["nano", "tiny", "small"])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--linear-chunk", type=int, default=None, help="override (mis. 64 vs 128)")
    ap.add_argument("--moe-impl", default=None, choices=["grouped", "loop"])
    ap.add_argument("--local-attn-impl", default=None, choices=["sdpa", "flex"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    dtype = torch.bfloat16
    over = {}
    if args.linear_chunk:
        over["linear_chunk"] = args.linear_chunk
    if args.moe_impl:
        over["moe_impl"] = args.moe_impl
    if args.local_attn_impl:
        over["local_attn_impl"] = args.local_attn_impl
    cfg = asca_config(args.preset, 4096, args.seq_len, **over)
    model = build_asca(args.preset, 4096, args.seq_len, **over).to(device).train()
    block = model.blocks[0]
    print(
        f"device={device} preset={args.preset} seq={args.seq_len} batch={args.batch} "
        f"chunk={cfg.linear_chunk} moe={cfg.moe_impl} local={cfg.local_attn_impl} "
        f"params={sum(p.numel() for p in model.parameters()):,}\n"
    )

    x = torch.randn(args.batch, args.seq_len, cfg.d_model, device=device)
    cos, sin = rope_cache(args.seq_len, cfg.head_dim, cfg.rope_theta, device, dtype)

    rows = []
    rows.append(bench("semantic_encoder", lambda: model.sem(x), list(model.sem.parameters()), args.iters))
    rows.append(
        bench("attention (lokal+linear)", lambda: block.attn(x, cos, sin), list(block.attn.parameters()), args.iters)
    )
    if getattr(block, "mem", None) is not None:
        rows.append(bench("memory", lambda: block.mem(x), list(block.mem.parameters()), args.iters))
    if hasattr(block.mlp, "experts"):
        rows.append(bench("moe", lambda: block.mlp(x)[0], list(block.mlp.parameters()), args.iters))
        rows.append(
            bench(
                "moe.shared saja",
                lambda: block.mlp.shared(x),
                list(block.mlp.shared.parameters()),
                args.iters,
            )
        )
        rows.append(
            bench(
                "moe.routed saja (tanpa shared)",
                lambda: block.mlp.experts[0](x),
                list(block.mlp.experts[0].parameters()),
                args.iters,
            )
        )
    rows.append(bench(f"ASCABlock (1/{cfg.n_layers})", lambda: block(x, cos, sin)[0], list(block.parameters()), args.iters))

    ids = torch.randint(0, 4096, (args.batch, args.seq_len), device=device)
    rows.append(bench("ASCAModel (total)", lambda: model(ids)[0], list(model.parameters()), args.iters))

    print(md_table(rows))
    total = rows[-1]["fwd+bwd_ms"]
    print(f"throughput fwd+bwd: {args.batch * args.seq_len / (total / 1000):,.0f} tokens/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
