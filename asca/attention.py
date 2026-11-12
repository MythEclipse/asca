"""Adaptive Hybrid Attention (docs/SPEC_ASCA.md §3.2).

Dua jalur paralel:
  A. local causal attention dengan sliding window ``w`` + RoPE   -> O(T*w*D)
  B. gated linear attention (chunked, paralel) dengan forget gate -> O(T*Dh^2)

Digabung dengan gate data-dependent per token per kanal:
    o = g * o_local + (1 - g) * o_global

**Decay per token.** Forget gate dihitung per token, ``lambda_t = sigmoid(W_lambda x_t)``,
sehingga bentuk chunked saat training dan bentuk rekuren saat decode (``step``)
secara matematis identik — prasyarat untuk KV-cache. State yang dibawa saat decode:

* lokal  : cache ``k``/``v`` sepanjang ``w`` token terakhir (O(w*Dh), bukan O(T))
* linear : ``S in R^{Dh x Dh}`` + ``ksum in R^{Dh}`` per head (O(Dh^2))
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
            # forget gate per token: lambda_t = sigmoid(W_lambda x_t)
            self.decay = nn.Linear(self.d, self.h, bias=True)

    # --------------------------------------------------------------- forward
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
            loglam = self._log_decay(x)  # (B,H,T) <= 0
            o_glb = self._global(q, k, v, loglam, T)
            g = torch.sigmoid(self.gate(x))  # (B,T,D)
            g = g.view(B, T, self.h, self.dh).transpose(1, 2)
            out = g * o_loc + (1.0 - g) * o_glb
        else:
            out = o_loc

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)

    def _log_decay(self, x: torch.Tensor) -> torch.Tensor:
        """log lambda_t per head, selalu <= 0."""
        return F.logsigmoid(self.decay(x)).transpose(1, 2)  # (B,H,T)

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
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        loglam: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Gated linear attention bentuk chunked (loop hanya sebanyak T/C chunk).

        Bentuk rekuren yang diemulasi (per token, per head):

            S_t   = lam_t * S_{t-1} + phi(k_t) v_t^T
            ksum_t = lam_t * ksum_{t-1} + phi(k_t)
            o_t   = phi(q_t)^T S_t / (phi(q_t)^T ksum_t + eps)

        Bentuk chunked memakai bobot ``exp(Lambda_i - Lambda_j) <= 1`` sehingga
        semua faktor dibatasi dan tidak ada overflow.
        """
        B, H, _, Dh = q.shape
        C = self.effective_chunk(T)
        pad = (-T) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            loglam = F.pad(loglam, (0, pad))
        Tp = T + pad
        S = Tp // C

        qf = feature_map(q).view(B, H, S, C, Dh)
        kf = feature_map(k).view(B, H, S, C, Dh)
        vc = v.view(B, H, S, C, Dh)

        # Lambda_i = jumlah log lambda di dalam chunk (<= 0, menurun)
        cum = loglam.view(B, H, S, C).cumsum(dim=-1)

        # --- intra-chunk: (phi(Q) phi(K)^T * exp(Lambda_i - Lambda_j)) V
        attn = qf @ kf.transpose(-1, -2)  # (B,H,S,C,C)
        causal = torch.ones(C, C, device=q.device, dtype=torch.bool).tril()
        diff = cum.unsqueeze(-1) - cum.unsqueeze(-2)  # Lambda_i - Lambda_j
        # mask SEBELUM exp: exp(-inf) = 0 (bukan nan), jadi tidak ada overflow di j > i
        decay = torch.exp(diff.masked_fill(~causal, float("-inf")))
        attn = attn * decay
        o_intra = attn @ vc
        denom_intra = attn.sum(dim=-1, keepdim=True)  # (B,H,S,C,1)

        # --- inter-chunk: state rekuren dibawa antar chunk
        state = q.new_zeros(B, H, Dh, Dh)
        ksum = q.new_zeros(B, H, Dh)
        o_inter = q.new_zeros(B, H, S, C, Dh)
        denom_inter = q.new_zeros(B, H, S, C, 1)
        for s in range(S):
            cum_s = cum[:, :, s]  # (B,H,C) <= 0
            scale = cum_s.exp()  # bobot token ke akhir chunk
            o_inter[:, :, s] = (qf[:, :, s] @ state) * scale.unsqueeze(-1)
            denom_inter[:, :, s] = (qf[:, :, s] @ ksum.unsqueeze(-1)) * scale.unsqueeze(-1)
            w = (cum_s[:, :, -1:] - cum_s).exp()  # (B,H,C)
            kw = kf[:, :, s] * w.unsqueeze(-1)
            total = cum_s[:, :, -1].exp().view(B, H, 1, 1)
            state = total * state + kw.transpose(-1, -2) @ vc[:, :, s]
            ksum = total.view(B, H, 1) * ksum + kw.sum(dim=-2)

        out = (o_intra + o_inter) / (denom_intra + denom_inter + 1e-6)
        out = out.reshape(B, H, Tp, Dh)[:, :, :T]
        return out

    # ------------------------------------------------------------------ step
    def init_state(self, batch: int, device, dtype) -> dict:
        """Cache decode berukuran TETAP (ring buffer window + state linear).

        Ukuran tidak tumbuh mengikuti panjang konteks, sehingga (a) tidak ada
        alokasi baru per token dan (b) bentuk tensor stabil untuk ``torch.compile``.
        """
        w = self.window
        st = {
            "k": torch.zeros(batch, self.h, w, self.dh, device=device, dtype=dtype),
            "v": torch.zeros(batch, self.h, w, self.dh, device=device, dtype=dtype),
        }
        if self.use_global:
            st["S"] = torch.zeros(batch, self.h, self.dh, self.dh, device=device, dtype=dtype)
            st["ksum"] = torch.zeros(batch, self.h, self.dh, device=device, dtype=dtype)
        return st

    def step(self, x_t: torch.Tensor, pos: int, state: dict) -> torch.Tensor:
        """Satu token: ``x_t`` berbentuk ``(B, 1, D)``. Identik dengan ``forward``."""
        B, _, D = x_t.shape
        q, k, v = self.qkv(x_t).split(D, dim=-1)
        q = q.view(B, 1, self.h, self.dh).transpose(1, 2)
        k = k.view(B, 1, self.h, self.dh).transpose(1, 2)
        v = v.view(B, 1, self.h, self.dh).transpose(1, 2)

        cos, sin = rope_cache(pos + 1, self.dh, self.cfg.rope_theta, x_t.device, x_t.dtype)
        cos, sin = cos[pos : pos + 1], sin[pos : pos + 1]
        q_loc = apply_rope(q, cos, sin)
        k_loc = apply_rope(k, cos, sin)

        # --- lokal: tulis ke ring buffer pada slot pos % w, lalu baca berurutan
        w = self.window
        state["k"][:, :, pos % w] = k_loc.squeeze(2)
        state["v"][:, :, pos % w] = v.squeeze(2)
        j = torch.arange(w, device=x_t.device)
        absolute = pos + 1 - w + j  # posisi absolut tiap slot ring
        idx = absolute % w
        valid = absolute >= 0
        k_sel = state["k"].index_select(2, idx)
        v_sel = state["v"].index_select(2, idx)
        scores = (q_loc @ k_sel.transpose(-1, -2)) / math.sqrt(self.dh)
        scores = scores.masked_fill(~valid.view(1, 1, 1, w), float("-inf"))
        o_loc = torch.softmax(scores, dim=-1) @ v_sel

        if not self.use_global:
            out = o_loc
        else:
            loglam = self._log_decay(x_t)  # (B,H,1)
            lam = loglam.exp()
            qf = feature_map(q).squeeze(2)  # (B,H,Dh)
            kf = feature_map(k).squeeze(2)
            vv = v.squeeze(2)
            state["S"] = lam.view(B, self.h, 1, 1) * state["S"] + kf.unsqueeze(
                -1
            ) @ vv.unsqueeze(-2)
            state["ksum"] = lam.view(B, self.h, 1) * state["ksum"] + kf
            num = (qf.unsqueeze(-2) @ state["S"]).squeeze(-2)  # (B,H,Dh)
            den = (qf * state["ksum"]).sum(-1, keepdim=True)  # (B,H,1)
            o_glb = (num / (den + 1e-6)).unsqueeze(2)  # (B,H,1,Dh)
            g = torch.sigmoid(self.gate(x_t)).view(B, 1, self.h, self.dh).transpose(1, 2)
            out = g * o_loc + (1.0 - g) * o_glb

        out = out.transpose(1, 2).reshape(B, 1, D)
        return self.proj(out)
