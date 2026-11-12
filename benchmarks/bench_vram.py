#!/usr/bin/env python3
"""Benchmark VRAM & latensi decode pada berbagai panjang sekuens.

Mengukur:
  * peak VRAM untuk satu langkah training (fwd+bwd, bf16) pada T = 512..4096
  * latensi decode (ms/token) — ASCA belum punya KV-cache, jadi ini batas atas
    biaya: recompute penuh per token. Angka ini yang harus diperbaiki agar klaim
    "inference cepat" valid.

Contoh:
    env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/bench_vram.py --preset nano
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
    count_params,
    dense_matched_to,
    get_device,
    md_table,
    set_seed,
)


def peak_train_vram(model, vocab: int, seq_len: int, device) -> float | None:
    """Peak VRAM (MB) satu langkah fwd+bwd; ``None`` bila OOM."""
    try:
        model.to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randint(0, vocab, (1, seq_len), device=device)
        y = torch.randint(0, vocab, (1, seq_len), device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
            aux = getattr(model, "aux_loss", None)
            total = loss if aux is None else loss + aux
        total.backward()
        opt.step()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1e6
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


@torch.no_grad()
def decode_latency_ms(model, vocab: int, prompt_len: int, new_tokens: int, device) -> float:
    model.to(device).eval()
    x = torch.randint(0, vocab, (1, prompt_len), device=device)
    model.generate(x, max_new_tokens=3, temperature=1.0)  # warmup
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    model.generate(x, max_new_tokens=new_tokens, temperature=1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / new_tokens * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano", choices=["nano", "tiny", "small"])
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--seq-lens", default="512,1024,2048,4096")
    ap.add_argument("--decode-new-tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    if device.type != "cuda":
        print("Benchmark VRAM butuh CUDA.")
        return 1
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    max_seq = max(seq_lens)
    print(f"device={device} preset={args.preset} seq_lens={seq_lens}\n")

    rows = []
    for label, factory in (
        ("B0-dense", lambda: build_dense(args.preset, args.vocab, max_seq)),
        ("ASCA-full", lambda: build_asca(args.preset, args.vocab, max_seq)),
    ):
        model = factory()
        row = {"model": label, "params": count_params(model)}
        for T in seq_lens:
            vram = peak_train_vram(model, args.vocab, T, device)
            row[f"train_vram_T{T}"] = "OOM" if vram is None else round(vram, 1)
            print(f"  {label:12s} T={T:5d} train_vram={'OOM' if vram is None else f'{vram:.1f} MB'}", flush=True)
        row["decode_ms_per_tok"] = round(
            decode_latency_ms(model, args.vocab, min(256, max_seq), args.decode_new_tokens, device), 2
        )
        rows.append(row)
        del model
        torch.cuda.empty_cache()

    # dense yang parameter-nya disetarakan (pembanding paling adil)
    asca = build_asca(args.preset, args.vocab, max_seq)
    matched = dense_matched_to(asca, args.preset, max_seq)
    del asca
    row = {"model": "B0-dense-matched", "params": count_params(matched)}
    for T in seq_lens:
        vram = peak_train_vram(matched, args.vocab, T, device)
        row[f"train_vram_T{T}"] = "OOM" if vram is None else round(vram, 1)
        print(f"  {'dense-matched':12s} T={T:5d} train_vram={'OOM' if vram is None else f'{vram:.1f} MB'}", flush=True)
    row["decode_ms_per_tok"] = round(
        decode_latency_ms(matched, args.vocab, min(256, max_seq), args.decode_new_tokens, device), 2
    )
    rows.append(row)

    print("\n" + md_table(rows))
    if args.out:
        append_experiment(
            args.out,
            f"VRAM & latensi decode ({args.preset})",
            rows,
            {"preset": args.preset, "seq_lens": seq_lens, "vocab": args.vocab, "seed": args.seed, "device": str(device)},
        )
        print(f"Hasil ditambahkan ke {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
