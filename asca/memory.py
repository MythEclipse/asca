"""Efficient Memory — SegmentMemory (docs/SPEC_ASCA.md §3.4).

Sekuens dibagi menjadi segmen panjang ``L_s``. Setiap segmen diringkas menjadi
``M`` slot lewat cross-attention dengan query yang dipelajari. Token di segmen
``s`` hanya boleh membaca ringkasan segmen ``< s`` (kausal antar-segmen), sehingga
tidak ada kebocoran token masa depan.

Biaya: O(T*M*D + S^2*M*D) dengan S = T/L_s, jauh di bawah attention penuh O(T^2*D).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ASCAConfig


class SegmentMemory(nn.Module):
    def __init__(self, cfg: ASCAConfig) -> None:
        super().__init__()
        self.d = cfg.d_model
        self.n_slots = cfg.n_memory_slots
        self.seg = cfg.mem_segment_len
        self.scale = 1.0 / math.sqrt(cfg.d_model)

        self.q_mem = nn.Parameter(torch.randn(self.n_slots, self.d) * 0.02)
        self.k_proj = nn.Linear(self.d, self.d, bias=False)
        self.v_proj = nn.Linear(self.d, self.d, bias=False)
        self.q_proj = nn.Linear(self.d, self.d, bias=False)
        self.o_proj = nn.Linear(self.d, self.d, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        L = min(self.seg, T)
        pad = (-T) % L
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        Tp = T + pad
        S = Tp // L
        if S < 2:
            # hanya satu segmen: tidak ada masa lalu yang bisa dibaca
            return torch.zeros_like(x[:, :T])

        xr = x.view(B, S, L, D)
        k = self.k_proj(xr)  # (B,S,L,D)
        v = self.v_proj(xr)

        # ringkasan segmen: (B,S,M,D)
        attn = torch.softmax(self.q_mem.view(1, 1, self.n_slots, D) @ k.transpose(-1, -2) * self.scale, dim=-1)
        summ = attn @ v

        # token di segmen s membaca ringkasan segmen < s
        q = self.q_proj(xr)  # (B,S,L,D)
        scores = torch.einsum("bsld,btmd->bstlm", q, summ) * self.scale  # (B,S,S,L,M)
        seg_idx = torch.arange(S, device=x.device)
        allow = seg_idx[None, :] < seg_idx[:, None]  # (S,S) -> True jika t < s
        # -inf tidak dipakai: baris tanpa entri valid (segmen 0) akan menghasilkan NaN
        # pada softmax. Pakai nilai sangat negatif lalu nolkan keluaran segmen 0.
        neg = torch.finfo(scores.dtype).min / 4
        scores = scores.masked_fill(~allow[None, :, :, None, None], neg)
        # Softmax HARUS dinormalisasi bersama atas (segmen t, slot m) — bukan per t,
        # kalau tidak baris yang seluruhnya ter-mask (uniform) ikut menyumbang dan
        # token bisa membaca segmennya sendiri (kebocoran kausal).
        w = torch.softmax(scores.reshape(B, S, S * L * self.n_slots), dim=-1)
        w = w.view(B, S, S, L, self.n_slots)
        out = torch.einsum("bstlm,btmd->bsld", w, summ)
        out = out * (seg_idx > 0).view(1, S, 1, 1)
        out = out.reshape(B, Tp, D)[:, :T]
        return torch.sigmoid(self.gate) * self.o_proj(out)
