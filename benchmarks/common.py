"""Utilitas bersama untuk benchmark & training.

Aturan proyek: perbandingan ASCA vs baseline harus *apple-to-apple* —
jumlah token latih, batch efektif, LR schedule, dan presisi identik.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from asca import ASCAConfig, ASCAModel  # noqa: E402
from baseline.dense_transformer import DenseConfig, DenseTransformer  # noqa: E402

#: Preset skala yang muat di RTX 4050 6 GB.
PRESETS: dict[str, dict] = {
    "nano": dict(d_model=128, n_layers=4, n_heads=4, max_seq_len=256),
    "tiny": dict(d_model=256, n_layers=6, n_heads=8, max_seq_len=512),
    "small": dict(d_model=384, n_layers=6, n_heads=6, max_seq_len=1024),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asca_config(
    preset: str = "tiny", vocab_size: int = 4096, seq_len: int | None = None, **over
) -> ASCAConfig:
    """Bangun ``ASCAConfig`` untuk preset tertentu (dipakai juga oleh bench ablasi)."""
    base = dict(PRESETS[preset])
    if seq_len is not None:
        base["max_seq_len"] = seq_len
    base["vocab_size"] = vocab_size
    # default yang masuk akal untuk semua preset
    base.setdefault("local_window", 128)
    base.setdefault("linear_chunk", 64)
    base.setdefault("mem_segment_len", 256)
    base.setdefault("n_memory_slots", 32)
    base.setdefault("n_experts", 8)
    base.setdefault("top_k", 2)
    base.update(over)
    return ASCAConfig(**base)


def build_asca(
    preset: str = "tiny", vocab_size: int = 4096, seq_len: int | None = None, **over
) -> ASCAModel:
    return ASCAModel(asca_config(preset, vocab_size, seq_len, **over))


def build_dense(
    preset: str = "tiny",
    vocab_size: int = 4096,
    seq_len: int | None = None,
    d_ff: int | None = None,
    **over,
) -> DenseTransformer:
    base = dict(PRESETS[preset])
    if seq_len is not None:
        base["max_seq_len"] = seq_len
    base["vocab_size"] = vocab_size
    if d_ff is not None:
        base["d_ff"] = d_ff
    base.update(over)
    return DenseTransformer(DenseConfig(**base))


def dense_matched_to(asca_model: ASCAModel, preset: str, seq_len: int) -> DenseTransformer:
    """Cari ``d_ff`` dense yang jumlah parameternya paling dekat dengan ASCA.

    Dibandingkan terhadap *active params per token* ASCA (kapasitas yang benar-benar
    dipakai saat inference), karena itu pembanding yang jujur untuk MoE.
    """
    target = asca_model.active_params_per_token()
    lo, hi = 32, 32 * asca_model.cfg.d_model
    best = None
    for _ in range(24):
        mid = (lo + hi) // 2
        mid = (mid + 7) // 8 * 8
        cand = build_dense(
            preset, vocab_size=asca_model.cfg.vocab_size, seq_len=seq_len, d_ff=mid
        )
        n = cand.active_params_per_token()
        if best is None or abs(n - target) < abs(best[0] - target):
            best = (n, mid)
        if n < target:
            lo = mid + 8
        else:
            hi = mid - 8
        if lo >= hi:
            break
    assert best is not None
    return build_dense(
        preset, vocab_size=asca_model.cfg.vocab_size, seq_len=seq_len, d_ff=best[1]
    )


# ------------------------------------------------------------------- metrik
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def maybe_compile(model: torch.nn.Module, enabled: bool = False) -> torch.nn.Module:
    """``torch.compile`` opsional.

    Terukur pada RTX 4050 (preset tiny, T=512): ASCA 61k -> 95k tok/s (1.54×),
    dense 166k -> 212k tok/s (1.28×). Biaya: ~10-15 s compile di langkah pertama.
    """
    if not enabled:
        return model
    return torch.compile(model)


def active_params(model: torch.nn.Module) -> int:
    if hasattr(model, "active_params_per_token"):
        return model.active_params_per_token()
    return count_params(model)


def count_flops_per_token(model: torch.nn.Module, seq_len: int, vocab_size: int) -> tuple[int, int]:
    """FLOPs terukur (forward) untuk satu sekuens ``seq_len``, dan per token.

    Memakai ``torch.utils.flop_counter`` sehingga yang dihitung adalah matmul/conv
    yang benar-benar dieksekusi (termasuk sparse routing MoE), bukan estimasi rumus.
    """
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    device = next(model.parameters()).device
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    total = counter.get_total_flops()
    return int(total), int(total // seq_len)


@dataclass
class TrainStats:
    model: str
    steps: int
    tokens: int
    train_loss: float
    val_loss: float
    ppl: float
    secs: float
    tokens_per_sec: float
    peak_vram_mb: float
    params: int
    active_params: int
    flops_per_token: int

    def as_row(self) -> dict:
        return {
            "model": self.model,
            "params": self.params,
            "active_params": self.active_params,
            "flops_per_token": self.flops_per_token,
            "tokens": self.tokens,
            "steps": self.steps,
            "train_loss": round(self.train_loss, 4),
            "val_loss": round(self.val_loss, 4),
            "ppl": round(self.ppl, 3),
            "secs": round(self.secs, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
        }


@dataclass
class TrainConfig:
    steps: int = 100
    batch_size: int = 8
    seq_len: int = 256
    lr: float = 3e-3
    weight_decay: float = 0.1
    warmup: int = 10
    grad_clip: float = 1.0
    log_every: int = 20
    eval_batches: int = 10
    aux_coef: float = 1.0
    amp_dtype: str = "bf16"
    warmup_steps: int = 2
    seed: int = 0


@torch.no_grad()
def evaluate(model, dataset, cfg: TrainConfig, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = dataset.get_batch("val", cfg.batch_size, cfg.seq_len, device)
        with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=device.type == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def _amp_dtype(cfg: TrainConfig) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.amp_dtype]


def train_model(
    model: torch.nn.Module,
    dataset,
    cfg: TrainConfig,
    device: torch.device,
    name: str,
    vocab_size: int,
) -> TrainStats:
    """Loop training minimal: AdamW + cosine + grad clip + bf16 autocast."""
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    total_tokens = cfg.steps * cfg.batch_size * cfg.seq_len
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, cfg.warmup))
        * 0.5
        * (1 + np.cos(np.pi * min(1.0, s / max(1, cfg.steps)))),
    )

    # warmup: memicu torch.compile & alokasi awal SEBELUM timer mulai, supaya
    # throughput tidak tercemar waktu kompilasi (~10-15 s).
    for _ in range(max(1, cfg.warmup_steps)):
        x, y = dataset.get_batch("train", cfg.batch_size, cfg.seq_len, device)
        with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=device.type == "cuda"):
            _, wloss = model(x, y)
        opt.zero_grad(set_to_none=True)
        wloss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    losses: list[float] = []
    for step in range(cfg.steps):
        x, y = dataset.get_batch("train", cfg.batch_size, cfg.seq_len, device)
        with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=device.type == "cuda"):
            _, loss = model(x, y)
            aux = getattr(model, "aux_loss", None)
            total = loss if aux is None else loss + cfg.aux_coef * aux
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            print(
                f"    [{name}] step {step + 1:4d}/{cfg.steps} loss {np.mean(losses[-cfg.log_every:]):.4f}"
                f" lr {sched.get_last_lr()[0]:.2e}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    secs = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    val_loss = evaluate(model, dataset, cfg, device)
    # FLOPs harus diukur pada modul asli: FlopCounterMode tidak menembus graph Inductor,
    # sehingga mengukur model ter-compile menghasilkan angka yang salah.
    raw_model = getattr(model, "_orig_mod", model)
    flops = count_flops_per_token(raw_model, cfg.seq_len, vocab_size)[1]
    return TrainStats(
        model=name,
        steps=cfg.steps,
        tokens=total_tokens,
        train_loss=float(np.mean(losses[-max(1, cfg.steps // 10) :])),
        val_loss=val_loss,
        ppl=float(np.exp(val_loss)),
        secs=secs,
        tokens_per_sec=total_tokens / max(secs, 1e-9),
        peak_vram_mb=peak,
        params=count_params(model),
        active_params=active_params(model),
        flops_per_token=flops,
    )


# -------------------------------------------------------------------- output
def aggregate(label: str, runs: list[dict]) -> dict:
    """Gabungkan beberapa seed menjadi satu baris ``mean ± std`` (protokol §6)."""
    import statistics as st

    def ms(key: str, fmt: str = "{:.4f}") -> str:
        vals = [float(r[key]) for r in runs]
        mean = sum(vals) / len(vals)
        sd = st.pstdev(vals) if len(vals) > 1 else 0.0
        return f"{fmt.format(mean)} ± {fmt.format(sd)}"

    first = runs[0]
    return {
        "model": label,
        "params": first["params"],
        "active_params": first["active_params"],
        "flops_per_token": first["flops_per_token"],
        "tokens": first["tokens"],
        "val_loss": ms("val_loss"),
        "ppl": ms("ppl", "{:.3f}"),
        "tokens_per_sec": ms("tokens_per_sec", "{:.0f}"),
        "peak_vram_mb": ms("peak_vram_mb", "{:.0f}"),
        "n_seeds": len(runs),
    }


def md_table(rows: Iterable[dict]) -> str:
    rows = list(rows)
    if not rows:
        return "_(tidak ada data)_\n"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def append_experiment(doc_path: str, title: str, rows: list[dict], meta: dict) -> None:
    """Tambahkan satu tabel hasil ke docs/EXPERIMENTS.md (append, bukan overwrite)."""
    os.makedirs(os.path.dirname(os.path.abspath(doc_path)), exist_ok=True)
    header = "" if os.path.exists(doc_path) else "# EXPERIMENTS — ASCA\n\nHasil benchmark otomatis. Setiap entri menyertakan konfigurasi lengkap.\n\n"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n## {title}\n\n"
        f"_{stamp} · seed {meta.get('seed')} · device {meta.get('device')} · "
        f"torch {torch.__version__}_\n\n"
        f"```json\n{json.dumps(meta, indent=2)}\n```\n\n"
        f"{md_table(rows)}\n"
    )
    with open(doc_path, "a", encoding="utf-8") as f:
        f.write(header + block)
