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
        self.max_seq = cfg.max_seq_len
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
        # Softmax per token: sumbu query (l) BUKAN bagian dari normalisasi, dan
        # normalisasi harus bersama atas (segmen t, slot m). Kalau l ikut masuk
        # sumbu softmax, penyebut tercampur antar posisi dan keluaran jadi salah.
        scores = scores.permute(0, 1, 3, 2, 4)  # (B,S,L,S,M)
        w = torch.softmax(scores.reshape(B, S, L, S * self.n_slots), dim=-1)
        w = w.view(B, S, L, S, self.n_slots)
        out = torch.einsum("bsltm,btmd->bsld", w, summ)
        out = out * (seg_idx > 0).view(1, S, 1, 1)
        out = out.reshape(B, Tp, D)[:, :T]
        return torch.sigmoid(self.gate) * self.o_proj(out)

    # ------------------------------------------------------------------ step
    def init_state(self, batch: int, device, dtype) -> dict:
        """Cache decode berukuran TETAP.

        ``summ`` dialokasikan untuk ``max_seg = max_seq_len // seg + 1`` segmen
        (bukan tumbuh per token), ``cur_*`` berukuran satu segmen penuh.
        """
        max_seg = self.max_seq // self.seg + 1
        return {
            "summ": torch.zeros(batch, max_seg, self.n_slots, self.d, device=device, dtype=dtype),
            "cur_k": torch.zeros(batch, self.seg, self.d, device=device, dtype=dtype),
            "cur_v": torch.zeros(batch, self.seg, self.d, device=device, dtype=dtype),
            "n_seg": 0,
            "n_cur": 0,
        }

    def step(self, x_t: torch.Tensor, pos: int, state: dict) -> torch.Tensor:
        """Satu token. Sama dengan ``forward`` pada sekuens penuh."""
        B, _, D = x_t.shape
        k_t = self.k_proj(x_t)  # (B,1,D)
        v_t = self.v_proj(x_t)

        # --- baca ringkasan segmen < segmen saat ini (belum termasuk token ini)
        n_seg = state["n_seg"]
        if n_seg == 0:
            out = torch.zeros_like(x_t)
        else:
            flat = state["summ"].reshape(B, -1, D)  # (B,max_seg*M,D)
            valid = torch.arange(flat.shape[1], device=x_t.device) < n_seg * self.n_slots
            q = self.q_proj(x_t)  # (B,1,D)
            scores = (q @ flat.transpose(-1, -2)) * self.scale  # (B,1,max_seg*M)
            scores = scores.masked_fill(~valid.view(1, 1, -1), float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out = w @ flat

        # --- akumulasi segmen aktif; saat penuh, ringkas & geser ke masa lalu
        p = state["n_cur"]
        state["cur_k"][:, p] = k_t.squeeze(1)
        state["cur_v"][:, p] = v_t.squeeze(1)
        state["n_cur"] = p + 1
        if state["n_cur"] == self.seg:
            ck, cv = state["cur_k"], state["cur_v"]  # (B,L,D)
            attn = torch.softmax(
                (self.q_mem.view(1, self.n_slots, D) @ ck.transpose(-1, -2)) * self.scale, dim=-1
            )  # (B,M,L)
            state["summ"][:, n_seg] = attn @ cv
            state["n_seg"] = n_seg + 1
            state["n_cur"] = 0

        return torch.sigmoid(self.gate) * self.o_proj(out)
