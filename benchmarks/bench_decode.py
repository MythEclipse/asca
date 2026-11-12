#!/usr/bin/env python3
"""Benchmark decode: latensi per token + memori cache, ASCA vs dense.

Membandingkan dua mode untuk tiap model:

* ``recompute`` — ``generate()``: tiap token menjalankan ulang seluruh konteks.
  Ini yang dilaporkan bench_vram.py sebagai batas atas (tanpa cache).
* ``cached``    — ``generate_cached()``: state rekuren (ASCA) / KV-cache penuh
  (dense), satu token per langkah.

Cache ASCA bersifat tetap ukurannya: window lokal O(w·Dh) + state linear O(Dh²)
+ ringkasan segmen O(S·M·D), jadi tidak tumbuh linier terhadap konteks seperti
KV-cache dense.

Contoh:
    env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_decode.py --preset tiny
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    build_asca,
    build_dense,
    dense_matched_to,
    get_device,
    md_table,
    set_seed,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def cache_bytes(cache: dict) -> int:
    """Total byte tensor di dalam struktur cache."""
    total = 0

    def walk(obj) -> None:
        nonlocal total
        if isinstance(obj, torch.Tensor):
            total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(cache)
    return total


def time_cached_decode(model, prompt_len: int, n_new: int, device) -> tuple[float, float, int]:
    """Kembalikan (prefill_tok_per_s, decode_ms_per_token, cache_bytes)."""
    idx = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
    model.eval()
    with torch.no_grad():
        cache = model.init_cache(1, device, model.embed.weight.dtype)
        t0 = time.perf_counter()
        logits = None
        for t in range(prompt_len):
            logits = model.forward_step(idx[:, t : t + 1], cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill = prompt_len / (time.perf_counter() - t0)

        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t0 = time.perf_counter()
        for _ in range(n_new):
            logits = model.forward_step(nxt, cache)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_tok = (time.perf_counter() - t0) / n_new * 1000.0
        size = cache_bytes(cache)
    return prefill, per_tok, size


def time_graph_decode(model, prompt_len: int, n_new: int, device) -> tuple[float, float]:
    """(prefill_tok_per_s, decode_ms_per_token) memakai CUDA graph."""
    from asca.graph_decode import GraphDecoder

    dec = GraphDecoder(model, batch=1)
    dec.reset()
    idx = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
    with torch.no_grad():
        t0 = time.perf_counter()
        for t in range(prompt_len):
            logits = dec.step(idx[:, t : t + 1])
        torch.cuda.synchronize()
        prefill = prompt_len / (time.perf_counter() - t0)

        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t0 = time.perf_counter()
        for _ in range(n_new):
            logits = dec.step(nxt)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        per_tok = (time.perf_counter() - t0) / n_new * 1000.0
    return prefill, per_tok


def time_recompute_decode(model, prompt_len: int, n_new: int, device) -> float:
    """Latensi per token untuk generate() tanpa cache (recompute penuh)."""
    idx = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
    model.eval()
    max_len = model.cfg.max_seq_len
    with torch.no_grad():
        for _ in range(2):  # warmup
            logits, _ = model(idx[:, -max_len:])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_new):
            logits, _ = model(idx[:, -max_len:])
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_new * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=["nano", "tiny", "small"])
    ap.add_argument("--contexts", default="512,1024,2048")
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-graph", action="store_true", help="lewati pengukuran CUDA graph")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    vocab = 4096
    contexts = [int(c) for c in args.contexts.split(",") if c]
    rows = []

    models = {}
    models["dense-matched"] = dense_matched_to(
        build_asca(args.preset, vocab, max(contexts)), args.preset, max(contexts)
    ).to(device)
    models["asca"] = build_asca(args.preset, vocab, max(contexts)).to(device)

    for name, model in models.items():
        for ctx in contexts:
            # sisakan ruang untuk token baru di dalam max_seq_len
            ctx = min(ctx, model.cfg.max_seq_len - args.new_tokens)
            if ctx <= 0:
                continue
            rec = time_recompute_decode(model, ctx, args.new_tokens, device)
            prefill, cached, size = time_cached_decode(model, ctx, args.new_tokens, device)
            g_prefill, g_cached = (None, None)
            if device.type == "cuda" and not args.no_graph:
                g_prefill, g_cached = time_graph_decode(model, ctx, args.new_tokens, device)
            rows.append(
                {
                    "model": name,
                    "context": ctx,
                    "recompute_ms_tok": round(rec, 2),
                    "cached_ms_tok": round(cached, 2),
                    "graph_ms_tok": None if g_cached is None else round(g_cached, 2),
                    "graph_vs_eager": None if g_cached is None else round(cached / g_cached, 2),
                    "graph_prefill_tok_s": None if g_prefill is None else round(g_prefill),
                    "cache_mb": round(size / 1024 / 1024, 2),
                }
            )

    table = md_table(rows)
    print(table)
    if args.out:
        with open(args.out, "a") as f:
            f.write(
                f"\n\n## Decode: latensi per token & memori cache ({args.preset}, "
                f"{args.new_tokens} token baru)\n\n{table}\n"
            )
        print(f"\nHasil ditambahkan ke {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
