#!/usr/bin/env python3
"""Training komparasi skala mini: ASCA vs Dense Transformer (budget token identik).

Ini eksperimen utama Tahap 1: apakah ASCA mencapai validation loss lebih rendah
daripada baseline dense pada **jumlah token latih, batch efektif, dan LR schedule
yang sama**. Hasil ditulis ke docs/EXPERIMENTS.md.

Contoh (smoke test cepat, CPU/GPU):
    env -u PYTHONPATH ~/.venvs/ml/bin/python benchmarks/train_toy.py \
        --preset nano --steps 100 --batch 8 --seq-len 256 --out docs/EXPERIMENTS.md

Contoh (lebih serius, GPU):
    ... --preset tiny --steps 2000 --batch 16 --seq-len 512 --corpus hf \
        --hf-repo HuggingFaceFW/fineweb --max-chars 50000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    TrainConfig,
    aggregate,
    append_experiment,
    asca_config,
    build_dense,
    dense_matched_to,
    get_device,
    maybe_compile,
    md_table,
    set_seed,
    train_model,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from asca import ASCAModel, ablation as make_ablation  # noqa: E402
from data.dataloader import build_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="nano", choices=["nano", "tiny", "small"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--aux-coef", type=float, default=1.0)
    ap.add_argument("--models", default="dense,asca", help="mis. dense,asca,A1,A3")
    ap.add_argument("--corpus", default="synthetic", choices=["synthetic", "file", "hf"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--hf-file", default=None)
    ap.add_argument("--max-chars", type=int, default=2_000_000)
    ap.add_argument("--vocab", type=int, default=None, help="default: ukuran tokenizer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="mis. 0,1,2 — agregasi mean±std (protokol: minimal 3 seed)")
    ap.add_argument("--compile", action="store_true", help="torch.compile model (lebih cepat, +10-15s di awal)")
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--set",
        default=None,
        help="override config ASCA, mis. 'n_experts=16,expert_ff=64' (dipisah koma)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    dataset = build_dataset(
        kind=args.corpus,
        path=args.path,
        hf_repo=args.hf_repo,
        hf_file=args.hf_file,
        max_chars=args.max_chars,
        seed=args.seed,
    )
    vocab = args.vocab or dataset.vocab_size
    print(
        f"device={device} corpus={args.corpus} tokens={len(dataset.tokens):,} vocab={vocab} "
        f"steps={args.steps} batch={args.batch} seq={args.seq_len} "
        f"(total {args.steps * args.batch * args.seq_len:,} token latih)\n"
    )

    tcfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch,
        seq_len=args.seq_len,
        lr=args.lr,
        warmup=args.warmup,
        log_every=args.log_every,
        eval_batches=args.eval_batches,
        aux_coef=args.aux_coef,
        seed=args.seed,
    )

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    over: dict = {}
    for item in (args.set or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, val = item.partition("=")
        key, val = key.strip(), val.strip()
        if val.lower() in ("true", "false"):
            over[key] = val.lower() == "true"
        else:
            try:
                over[key] = int(val)
            except ValueError:
                try:
                    over[key] = float(val)
                except ValueError:
                    over[key] = val
    if over:
        print(f"  override config: {over}", flush=True)
    base_cfg = asca_config(args.preset, vocab, args.seq_len, **over)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    rows = []
    raw: dict[str, list[dict]] = {}

    for name in wanted:
        for seed in seeds:
            set_seed(seed)  # inisialisasi & batch order identik untuk semua model
            if name == "dense":
                model = build_dense(args.preset, vocab, args.seq_len)
                label = "B0-dense"
            elif name == "dense-matched":
                asca_ref = ASCAModel(base_cfg)
                model = dense_matched_to(asca_ref, args.preset, args.seq_len)
                del asca_ref
                label = "B0-dense-matched"
            elif name in ("asca", "A0"):
                model = ASCAModel(base_cfg)
                label = "A0-asca"
            else:  # A1..A6
                model = ASCAModel(make_ablation(base_cfg, name))
                label = f"ASCA-{name}"
            print(
                f"  === {label} seed {seed} (params {sum(p.numel() for p in model.parameters()):,}) ===",
                flush=True,
            )
            model = maybe_compile(model, args.compile)
            stats = train_model(model, dataset, tcfg, device, f"{label}/s{seed}", vocab)
            raw.setdefault(label, []).append(stats.as_row())
            print(
                f"  -> {label}/s{seed}: val_loss={stats.val_loss:.4f} ppl={stats.ppl:.2f} "
                f"tok/s={stats.tokens_per_sec:.0f} vram={stats.peak_vram_mb:.0f}MB\n",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = [aggregate(label, runs) for label, runs in raw.items()]
    print(md_table(rows))
    if args.out:
        append_experiment(
            args.out,
            f"Training komparasi ({args.preset}, {args.steps} step{(' — ' + args.tag) if args.tag else ''})",
            rows,
            {
                "preset": args.preset,
                "corpus": args.corpus,
                "tokens_in_corpus": int(len(dataset.tokens)),
                "vocab": vocab,
                "steps": args.steps,
                "batch": args.batch,
                "seq_len": args.seq_len,
                "train_tokens": args.steps * args.batch * args.seq_len,
                "lr": args.lr,
                "warmup": args.warmup,
                "seed": args.seed,
                "device": str(device),
                "torch": torch.__version__,
                "asca_config": base_cfg.to_dict(),
                "models": wanted,
                "compile": bool(args.compile),
            },
        )
        print(f"Hasil ditambahkan ke {args.out}")
    if args.tag:
        print(json.dumps(raw, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
