"""Sparse Experts + Shared Expert (docs/SPEC_ASCA.md §3.3).

``AdaptiveSparseMoE`` = shared expert (selalu aktif) + top-k routed experts
dengan adaptive compute (threshold skip).

Dua implementasi dispatch (dipilih lewat ``ASCAConfig.moe_impl``):

* ``"grouped"`` (default) — *grouped GEMM*: token diurutkan per expert, di-pad
  menjadi blok ``(E, M, D)``, lalu **satu** ``bmm`` untuk seluruh expert.
  FLOPs tetap sparse (sebanding token yang benar-benar dirutekan), tetapi kernel
  launch turun dari ``3·E`` menjadi 2. Terukur ~1.8× lebih cepat dari ``loop``
  pada RTX 4050 (preset tiny, T=512).
* ``"loop"`` — referensi sederhana (gather/scatter per expert), dipakai sebagai
  pembanding kebenaran di ``tests/run_tests.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ASCAConfig
from .router import AdaptiveRouter


class SwiGLU(nn.Module):
    """MLP SwiGLU: ``W2(silu(W1 x) * W3 x)``."""

    def __init__(self, d_in: int, d_hidden: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=bias)
        self.w3 = nn.Linear(d_in, d_hidden, bias=bias)
        self.w2 = nn.Linear(d_hidden, d_in, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AdaptiveSparseMoE(nn.Module):
    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.impl = cfg.moe_impl
        self.router = AdaptiveRouter(cfg)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.expert_ff, cfg.bias) for _ in range(cfg.n_experts)]
        )
        self.shared = (
            SwiGLU(cfg.d_model, cfg.shared_ff, cfg.bias) if cfg.n_shared_experts > 0 else None
        )
        # statistik untuk logging (tidak mempengaruhi gradien)
        self.register_buffer("_expert_counts", torch.zeros(cfg.n_experts), persistent=False)
        self.register_buffer("_tokens_seen", torch.zeros((), dtype=torch.long), persistent=False)
        self._last_padding_factor = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat)
        weights, top_idx = self.router.route(logits)
        aux = self.router.aux_loss(logits, top_idx)

        # Dispatch: pada decode (token sangat sedikit) jalur grouped GEMM justru
        # didominasi overhead argsort/bincount/padding, dan `nonzero`/`.item()`
        # membuat jalur ini tidak bisa di-CUDA-graph. Pakai dispatch terpilih.
        n_pairs = flat.shape[0] * self.top_k
        if n_pairs <= 64:
            out = self._routed_selected(flat, weights, top_idx)
        elif self.impl == "grouped":
            out = self._routed_grouped(flat, weights, top_idx)
        else:
            out = self._routed_loop(flat, weights, top_idx)

        if self.shared is not None:
            out = out + self.shared(flat)

        if not self.training:
            with torch.no_grad():
                self._expert_counts += F.one_hot(top_idx, self.n_experts).sum(dim=(0, 1)).float()
                self._tokens_seen += flat.shape[0]

        return out.view(B, T, D), aux

    # ------------------------------------------------------------- dispatch
    def _routed_selected(
        self, flat: torch.Tensor, weights: torch.Tensor, top_idx: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch terpilih: hanya expert yang dipilih, lewat ``index_select``.

        Bentuk tensor tidak bergantung data (tanpa ``nonzero``/``.item()``), jadi
        jalur ini bisa di-CUDA-graph dan dipakai untuk decode. FLOPs tetap sparse:
        hanya ``N * top_k`` pasangan expert-token yang dihitung.
        """
        N = flat.shape[0]
        idx = top_idx.reshape(-1)  # (N*k,)
        w13 = torch.cat(
            [
                torch.stack([e.w1.weight for e in self.experts]),
                torch.stack([e.w3.weight for e in self.experts]),
            ],
            dim=1,
        )  # (E, 2*dh, D)
        w2 = torch.stack([e.w2.weight for e in self.experts])  # (E, D, dh)
        w13_sel = w13.index_select(0, idx)  # (N*k, 2dh, D)
        w2_sel = w2.index_select(0, idx)  # (N*k, D, dh)
        xr = flat.repeat_interleave(self.top_k, dim=0)  # (N*k, D)
        h = torch.bmm(w13_sel, xr.unsqueeze(-1)).squeeze(-1)  # (N*k, 2dh)
        gate, up = h.chunk(2, dim=-1)
        act = F.silu(gate) * up
        o = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)  # (N*k, D)
        o = o * weights.reshape(-1, 1)
        return o.view(N, self.top_k, self.cfg.d_model).sum(dim=1)

    def _routed_loop(
        self, flat: torch.Tensor, weights: torch.Tensor, top_idx: torch.Tensor
    ) -> torch.Tensor:
        """Referensi: satu matmul kecil per expert (banyak kernel launch)."""
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            h = expert(flat.index_select(0, tok))
            out.index_add_(0, tok, h * weights[tok, slot].unsqueeze(-1))
        return out

    def _routed_grouped(
        self, flat: torch.Tensor, weights: torch.Tensor, top_idx: torch.Tensor
    ) -> torch.Tensor:
        """Grouped GEMM: urutkan token per expert, satu ``bmm`` untuk semua expert."""
        N = flat.shape[0]
        device = flat.device
        tok = torch.arange(N, device=device).repeat_interleave(self.top_k)
        exp = top_idx.reshape(-1)
        wgt = weights.reshape(-1)

        order = torch.argsort(exp, stable=True)
        tok_s, exp_s, wgt_s = tok[order], exp[order], wgt[order]
        counts = torch.bincount(exp_s, minlength=self.n_experts)
        M = int(counts.max().item())
        if M == 0:
            return torch.zeros_like(flat)

        # posisi tiap pasangan di dalam blok expert-nya
        offs = torch.cumsum(counts, 0) - counts
        pos = torch.arange(exp_s.numel(), device=device) - offs[exp_s]

        # diagnostik: seberapa banyak padding yang dibayar (1.0 = routing sempurna rata)
        self._last_padding_factor = float(self.n_experts * M) / max(1, exp_s.numel())
        block = torch.zeros(self.n_experts, M, dtype=torch.long, device=device)
        block[exp_s, pos] = tok_s
        valid = torch.zeros(self.n_experts, M, dtype=torch.bool, device=device)
        valid[exp_s, pos] = True

        # gabungkan bobot semua expert -> 2 bmm total untuk seluruh expert
        w13 = torch.cat(
            [
                torch.stack([e.w1.weight for e in self.experts]),
                torch.stack([e.w3.weight for e in self.experts]),
            ],
            dim=1,
        )  # (E, 2*expert_ff, D)
        w2 = torch.stack([e.w2.weight for e in self.experts])  # (E, D, expert_ff)

        xg = flat[block]  # (E, M, D)
        h = torch.bmm(xg, w13.transpose(1, 2))
        gate, up = h.chunk(2, dim=-1)
        act = F.silu(gate) * up
        act = act * valid.unsqueeze(-1).to(act.dtype)  # slot padding -> nol
        o = torch.bmm(act, w2.transpose(1, 2))  # (E, M, D)

        wg = torch.zeros(self.n_experts, M, device=device, dtype=o.dtype)
        wg[exp_s, pos] = wgt_s.to(o.dtype)
        o = o * wg.unsqueeze(-1)

        out = torch.zeros(N, self.cfg.d_model, device=device, dtype=o.dtype)
        out.index_add_(0, block.reshape(-1), o.reshape(-1, self.cfg.d_model))
        return out

    # ------------------------------------------------------------- statistik
    @property
    def padding_factor(self) -> float:
        """Rasio FLOPs routed yang benar-benar dibayar vs yang ideal (grouped GEMM).

        1.0 = routing sempurna rata; 1.6 = satu expert menerima 60% token lebih
        banyak dari rata-rata sehingga blok padded ikut dihitung.
        """
        return self._last_padding_factor

    @torch.no_grad()
    def expert_usage(self) -> torch.Tensor:
        """Fraksi token yang dirutekan ke setiap expert sejak reset terakhir."""
        total = self._expert_counts.sum().clamp(min=1.0)
        return self._expert_counts / total

    @torch.no_grad()
    def reset_usage(self) -> None:
        self._expert_counts.zero_()
        self._tokens_seen.zero_()
