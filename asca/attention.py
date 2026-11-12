"""Adaptive Hybrid Attention (docs/SPEC_ASCA.md §3.2).

Dua jalur paralel:
  A. local causal attention dengan sliding window ``w`` + RoPE   -> O(T*w*D)
  B. gated linear attention (chunked, paralel) dengan forget gate -> O(T*Dh^2)

Digabung dengan gate data-dependent per token per kanal:
    o = g * o_local + (1 - g) * o_global
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ASCAConfig
from .norm import apply_rope, rope_cache

_WINDOW_MASK_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_FLEX_BLOCK_MASK_CACHE: dict[tuple[int, int, str], object] = {}
_FLEX_COMPILED = None


def _flex_available() -> bool:
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

        return True
    except Exception:  # pragma: no cover - versi torch lama
        return False


def _get_flex_compiled():
    """Compile ``flex_attention`` sekali per proses (eager-nya sangat lambat)."""
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
    return _FLEX_COMPILED


def flex_block_mask(t: int, w: int, device: torch.device):
    """BlockMask kausal berjendela ``w``, di-cache per (T, w, device)."""
    key = (t, w, str(device))
    cached = _FLEX_BLOCK_MASK_CACHE.get(key)
    if cached is None:
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, qi, ki):
            return (ki <= qi) & (ki > qi - w)

        cached = create_block_mask(mask_mod, B=None, H=None, Q_LEN=t, KV_LEN=t, device=device)
        _FLEX_BLOCK_MASK_CACHE[key] = cached
    return cached


def window_mask(t: int, w: int, device: torch.device) -> torch.Tensor:
    """Mask bool ``(t, t)``; ``True`` = boleh diperhatikan (``j <= i`` dan ``j > i - w``)."""
    key = (t, w, str(device))
    cached = _WINDOW_MASK_CACHE.get(key)
    if cached is None:
        i = torch.arange(t, device=device)[:, None]
        j = torch.arange(t, device=device)[None, :]
        cached = (j <= i) & (j > i - w)
        _WINDOW_MASK_CACHE[key] = cached
    return cached


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """phi(x) = elu(x) + 1, selalu positif (Katharopoulos et al., 2020)."""
    return F.elu(x) + 1.0


class AdaptiveHybridAttention(nn.Module):
    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.dh = cfg.head_dim
        self.d = cfg.d_model
        self.window = cfg.local_window if cfg.local_window > 0 else cfg.max_seq_len
        self.use_global = cfg.use_hybrid_attention
        self.chunk = cfg.linear_chunk

        self.qkv = nn.Linear(self.d, 3 * self.d, bias=cfg.bias)
        self.proj = nn.Linear(self.d, self.d, bias=cfg.bias)
        self.dropout = cfg.dropout

        if self.use_global:
            # gate penggabung: g = sigmoid(W_g x)
            self.gate = nn.Linear(self.d, self.d, bias=True)
            # forget gate per head dari rata-rata chunk: lambda = sigmoid(W_lambda xbar)
            self.decay = nn.Linear(self.d, self.h, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)

        if cos is None or sin is None:
            cos, sin = rope_cache(T, self.dh, self.cfg.rope_theta, x.device, x.dtype)
        q_loc = apply_rope(q, cos, sin)
        k_loc = apply_rope(k, cos, sin)

        o_loc = self._local(q_loc, k_loc, v, T)
        if self.use_global:
            o_glb = self._global(x, q, k, v, T)
            g = torch.sigmoid(self.gate(x))  # (B,T,D)
            g = g.view(B, T, self.h, self.dh).transpose(1, 2)
            out = g * o_loc + (1.0 - g) * o_glb
        else:
            out = o_loc

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)

    # ------------------------------------------------------------------ lokal
    def _local(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, T: int) -> torch.Tensor:
        w = min(self.window, T)
        drop = self.dropout if self.training else 0.0
        impl = self.cfg.local_attn_impl
        if w >= T:
            # jendela penuh == kausal biasa: pakai jalur cepat SDPA
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        if impl == "flex" and _flex_available():
            bm = flex_block_mask(T, w, q.device)
            return _get_flex_compiled()(q, k, v, block_mask=bm)
        mask = window_mask(T, w, q.device)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)

    # ----------------------------------------------------------------- global
    def effective_chunk(self, T: int) -> int:
        """Ukuran chunk efektif.

        ``linear_chunk`` besar = lebih sedikit iterasi loop (lebih cepat di GPU),
        tetapi biaya intra-chunk tumbuh O(T·C) dan VRAM naik O(C²). Karena itu
        chunk dibatasi ``T/4`` agar sifat sub-kuadratik tidak hilang pada T kecil.
        """
        return max(8, min(self.chunk, max(8, T // 4)))

    def _global(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Gated linear attention bentuk chunked (loop hanya sebanyak T/C chunk)."""
        B, H, _, Dh = q.shape
        C = self.effective_chunk(T)
        pad = (-T) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            x = F.pad(x, (0, 0, 0, pad))
        Tp = T + pad
        S = Tp // C

        qf = feature_map(q).view(B, H, S, C, Dh)
        kf = feature_map(k).view(B, H, S, C, Dh)
        vc = v.view(B, H, S, C, Dh)

        # forget gate per head, konstan di dalam chunk
        xbar = x.reshape(B, S, C, -1).mean(dim=2)  # (B,S,D)
        lam = torch.sigmoid(self.decay(xbar)).transpose(1, 2)  # (B,H,S)

        # --- intra-chunk: (phi(Q) phi(K)^T o tril) V
        attn = qf @ kf.transpose(-1, -2)  # (B,H,S,C,C)
        causal = torch.ones(C, C, device=q.device, dtype=torch.bool).tril()
        attn = attn.masked_fill(~causal, 0.0)
        o_intra = attn @ vc
        denom_intra = attn.sum(dim=-1, keepdim=True)  # (B,H,S,C,1)

        # --- inter-chunk: state rekuren S_c = lam_c * S_{c-1} + phi(K_c)^T V_c
        state = q.new_zeros(B, H, Dh, Dh)
        ksum = q.new_zeros(B, H, Dh)
        o_inter = q.new_zeros(B, H, S, C, Dh)
        denom_inter = q.new_zeros(B, H, S, C, 1)
        for s in range(S):
            o_inter[:, :, s] = qf[:, :, s] @ state
            denom_inter[:, :, s] = (qf[:, :, s] @ ksum.unsqueeze(-1))
            lam_s = lam[:, :, s].view(B, H, 1, 1)
            state = state * lam_s + kf[:, :, s].transpose(-1, -2) @ vc[:, :, s]
            ksum = ksum * lam[:, :, s].view(B, H, 1) + kf[:, :, s].sum(dim=-2)

        out = (o_intra + o_inter) / (denom_intra + denom_inter + 1e-6)
        out = out.reshape(B, H, Tp, Dh)[:, :, :T]
        return out
