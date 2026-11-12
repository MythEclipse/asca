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
        # q/k/v difusikan jadi satu GEMM (inputnya sama); o_proj terpisah
        self.in_proj = nn.Linear(self.d, 3 * self.d, bias=False)
        self.o_proj = nn.Linear(self.d, self.d, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_state(x)[0]

    def forward_with_state(self, x: torch.Tensor):
        """Seperti ``forward`` tetapi juga mengembalikan state cache decode.

        State yang dikembalikan: ringkasan segmen yang SUDAH selesai
        (``n_seg = T // L_s``) plus akumulator online-softmax untuk segmen aktif.
        """
        B, T, D = x.shape
        L = min(self.seg, T)
        pad = (-T) % L
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        Tp = T + pad
        S = Tp // L
        if S < 2:
            # hanya satu segmen: tidak ada masa lalu yang bisa dibaca
            return torch.zeros_like(x[:, :T]), self._state_from_prefix(x, T)

        xr = x.view(B, S, L, D)
        qkv = self.in_proj(xr)  # (B,S,L,3D)
        qk, k, v = qkv.split(D, dim=-1)

        # ringkasan segmen: (B,S,M,D)
        attn = torch.softmax(self.q_mem.view(1, 1, self.n_slots, D) @ k.transpose(-1, -2) * self.scale, dim=-1)
        summ = attn @ v

        # token di segmen s membaca ringkasan segmen < s
        q = qk  # (B,S,L,D)
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
        return torch.sigmoid(self.gate) * self.o_proj(out), self._state_from_prefix(x[:, :T], T, summ)

    def _state_from_prefix(self, x: torch.Tensor, T: int, summ: torch.Tensor | None = None) -> dict:
        """Bangun state decode dari lintasan paralel untuk prefiks sepanjang ``T``."""
        B, _, D = x.shape
        max_seg = self.max_seq // self.seg + 1
        n_seg = min(T // self.seg, max_seg - 1)
        n_cur = T - n_seg * self.seg

        state = {
            "summ": torch.zeros(B, max_seg, self.n_slots, D, device=x.device, dtype=x.dtype),
            "acc": torch.zeros(B, self.n_slots, D, device=x.device, dtype=x.dtype),
            "l": torch.zeros(B, self.n_slots, device=x.device, dtype=x.dtype),
            "m": torch.full((B, self.n_slots), float("-inf"), device=x.device, dtype=x.dtype),
            "n_seg": torch.tensor(n_seg, dtype=torch.long, device=x.device),
            "n_cur": torch.tensor(n_cur, dtype=torch.long, device=x.device),
        }
        if summ is not None and n_seg > 0:
            state["summ"][:, :n_seg] = summ[:, :n_seg]

        if n_cur > 0:
            # akumulator online-softmax untuk token segmen aktif yang belum selesai
            part = x[:, n_seg * self.seg :]
            _, k, v = self.in_proj(part).split(D, dim=-1)  # (B,n_cur,D)
            sc = (self.q_mem.view(1, self.n_slots, D) @ k.transpose(-1, -2)) * self.scale  # (B,M,n_cur)
            m = sc.max(dim=-1).values
            p = torch.exp(sc - m.unsqueeze(-1))
            state["acc"] = p @ v
            state["l"] = p.sum(dim=-1)
            state["m"] = m
        return state

    # ------------------------------------------------------------------ step
    def init_state(self, batch: int, device, dtype) -> dict:
        """Cache decode berukuran TETAP dan bebas cabang (CUDA-graph friendly).

        Ringkasan segmen aktif dihitung inkremental dengan online softmax
        (flash-style): ``acc`` (pembilang), ``l`` (penyebut), ``m`` (maks berjalan).
        Biaya per token jadi O(M·D), bukan O(M·L_s·D).
        """
        max_seg = self.max_seq // self.seg + 1
        return {
            "summ": torch.zeros(batch, max_seg, self.n_slots, self.d, device=device, dtype=dtype),
            "acc": torch.zeros(batch, self.n_slots, self.d, device=device, dtype=dtype),
            "l": torch.zeros(batch, self.n_slots, device=device, dtype=dtype),
            "m": torch.full((batch, self.n_slots), float("-inf"), device=device, dtype=dtype),
            "n_seg": torch.zeros((), dtype=torch.long, device=device),
            "n_cur": torch.zeros((), dtype=torch.long, device=device),
        }

    def step(self, x_t: torch.Tensor, pos_t: torch.Tensor, rope, state: dict) -> torch.Tensor:
        """Satu token. Sama dengan ``forward`` pada sekuens penuh."""
        B, _, D = x_t.shape
        q_t, k_t, v_t = self.in_proj(x_t).split(D, dim=-1)  # (B,1,D)

        # --- baca ringkasan segmen < segmen saat ini (belum termasuk token ini)
        flat = state["summ"].reshape(B, -1, D)  # (B,max_seg*M,D)
        n_valid = state["n_seg"] * self.n_slots
        valid = torch.arange(flat.shape[1], device=x_t.device) < n_valid
        scores = (q_t @ flat.transpose(-1, -2)) * self.scale  # (B,1,max_seg*M)
        # -1e30 (bukan -inf) agar aman untuk CUDA graph. Kalau belum ada segmen,
        # semua ter-mask -> bobot uniform, dan karena `summ` masih nol keluarannya
        # tepat nol (sama dengan forward).
        scores = scores.masked_fill(~valid.view(1, 1, -1), -1e30)
        w = torch.softmax(scores, dim=-1)
        out = w @ flat

        # --- ringkasan segmen aktif secara inkremental (online softmax).
        # SEMUA update state in-place: kalau buffer di-rebind ke tensor baru saat
        # CUDA-graph capture, memori lama dilepas di tengah capture dan graph bisa
        # menulis ke blok yang masih dipakai tensor lain (aliasing -> data rusak).
        sc = (self.q_mem.view(1, self.n_slots, D) @ k_t.transpose(-1, -2)).squeeze(-1) * self.scale
        m_new = torch.maximum(state["m"], sc)
        alpha = torch.exp(state["m"] - m_new)  # m lama -inf -> 0
        p = torch.exp(sc - m_new)
        state["l"].mul_(alpha).add_(p)
        state["acc"].mul_(alpha.unsqueeze(-1)).add_(p.unsqueeze(-1) * v_t)
        state["m"].copy_(m_new)

        # --- saat segmen penuh: tulis ringkasan ke `summ`, reset akumulator
        n_seg = state["n_seg"].clamp(max=state["summ"].shape[1] - 1)
        summary = state["acc"] / state["l"].clamp(min=1e-20).unsqueeze(-1)  # (B,M,D)
        old = state["summ"].index_select(1, n_seg.view(1))  # (B,1,M,D)
        is_full = (state["n_cur"] + 1) >= self.seg
        new = torch.where(is_full, summary.unsqueeze(1), old)
        state["summ"].scatter_(1, n_seg.view(1, 1, 1, 1).expand(B, 1, self.n_slots, D), new)
        state["n_seg"].add_(is_full.long())
        state["n_cur"].add_(1).masked_fill_(is_full, 0)
        state["acc"].masked_fill_(is_full.unsqueeze(-1), 0.0)
        state["l"].masked_fill_(is_full, 0.0)
        state["m"].masked_fill_(is_full, float("-inf"))

        return torch.sigmoid(self.gate) * self.o_proj(out)
