"""ASCA v2 — gated delta-rule state block: attention-free, constant-state.

Inti arsitektur v2. Tidak ada attention, tidak ada softmax, tidak ada KV-cache yang
tumbuh. Setiap head menyimpan state asosiatif ``S in R^{d_k x d_v}`` yang diperbarui
dengan **delta rule** (Schlag et al., 2021; gated varian Yang et al., 2024):

    S_t = alpha_t * S_{t-1} + beta_t * k_t (v_t - alpha_t * S_{t-1}^T k_t)^T
    o_t = S_t^T q_t

Suku ``(v_t - S^T k_t)`` adalah kuncinya: state tidak menumpuk buta seperti linear
attention, tetapi menulis ulang slot yang sudah ada (memori asosiatif dengan
overwrite). ``alpha_t`` adalah forget gate per token, ``beta_t`` laju tulis.

Dua bentuk implementasi, keduanya harus identik secara numerik:

* :func:`delta_recurrent` — O(1) memori per token, dipakai saat decode.
* :func:`delta_chunked`  — bentuk paralel per chunk (UT-transform), dipakai saat
  training supaya O(T) dan tidak butuh loop token-per-token.

Paritas keduanya diuji di ``tests/test_delta.py``; ini gerbang wajib sebelum kode
diport ke kernel Rust/CUDA.
"""

from __future__ import annotations

import torch

__all__ = ["delta_recurrent", "delta_chunked", "GatedDeltaMemory"]


# --------------------------------------------------------------------- rekuren
def delta_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Bentuk rekuren (decode). ``q,k,v``: (B,H,T,D); ``alpha,beta``: (B,H,T)."""
    B, H, T, D = q.shape
    S = q.new_zeros(B, H, D, D)
    out = q.new_zeros(B, H, T, D)
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        a = alpha[:, :, t].unsqueeze(-1)  # (B,H,1)
        b = beta[:, :, t].unsqueeze(-1)
        # S <- a*S + b*k*(v - a*S^T k)^T
        Sk = torch.einsum("bhd,bhde->bhe", kt, S)  # S^T k  (B,H,D)
        u = b * (vt - a * Sk)  # (B,H,D)
        S = a.unsqueeze(-1) * S + kt.unsqueeze(-1) * u.unsqueeze(-2)
        out[:, :, t] = torch.einsum("bhd,bhde->bhe", qt, S)
    return out


# --------------------------------------------------------------------- chunked
def delta_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    chunk: int = 64,
) -> torch.Tensor:
    """Bentuk paralel per chunk. Sama persis dengan :func:`delta_recurrent`.

    Turunan (per chunk, ``gamma_t = prod_{i<=t} alpha_i`` di dalam chunk):

        u_j + beta_j*gamma_j * sum_{i<j} (1/gamma_i)(k_i.k_j) u_i = beta_j (v_j - gamma_j S_0^T k_j)
        U      = (I + L)^-1 diag(beta) (V - gamma * (K S_0))
        S_end  = gamma_C S_0 + Ktilde^T (gamma_C U)
        O      = gamma * (Q S_0 + tril(Q Ktilde^T, 0) U)

    dengan ``Ktilde`` = baris K diskalakan ``1/gamma`` dan ``L[i,j] = beta_j gamma_j
    (1/gamma_i)(k_i.k_j)`` untuk ``i < j``.
    """
    B, H, T, D = q.shape
    pad = (-T) % chunk
    if pad:
        q, k, v = (torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        alpha = torch.nn.functional.pad(alpha, (0, pad), value=1.0)
        beta = torch.nn.functional.pad(beta, (0, pad), value=0.0)
    Tp = T + pad
    nc = Tp // chunk

    q, k, v = (x.view(B, H, nc, chunk, D) for x in (q, k, v))
    a = alpha.view(B, H, nc, chunk)
    b = beta.view(B, H, nc, chunk)

    # Decay RELATIF: rho_t = gamma_t / gamma_C, dengan gamma_C = produk seluruh chunk.
    # Ini kunci kestabilan numerik: rho >= 1 sehingga 1/rho <= 1, sedangkan gamma
    # mentah bisa sekecil 1e-40 pada chunk panjang dan membuat 1/gamma meledak.
    gamma = torch.cumprod(a, dim=-1)  # (B,H,nc,C)
    g_last = gamma[..., -1:].clamp_min(1e-30)  # (B,H,nc,1)
    rho = (gamma / g_last).clamp_min(1e-6)  # >= 1, dipakai untuk rasio di dalam chunk
    r_inv = 1.0 / rho
    k_t = k * r_inv.unsqueeze(-1)  # Ktilde: baris i diskalakan 1/rho_i

    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    # L[j,i] = beta_j*gamma_j * (k_j . k_i / gamma_i) untuk i < j
    #   -> pakai (K Ktilde^T)[j,i] lalu skala baris j dengan beta_j*gamma_j
    kk = torch.matmul(k, k_t.transpose(-1, -2))  # (B,H,nc,C,C)
    lower = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=torch.bool), -1)
    L = kk * (b * rho).unsqueeze(-1)
    L = torch.where(lower, L, torch.zeros_like(L))
    A = eye + L  # (I + L), lower-triangular satuan -> mudah di-inverse

    S = q.new_zeros(B, H, D, D)
    out = q.new_zeros(B, H, nc, chunk, D)
    for c in range(nc):
        qc, kc, vc = q[:, :, c], k[:, :, c], v[:, :, c]
        kt_c, r_c, b_c, A_c = k_t[:, :, c], rho[:, :, c], b[:, :, c], A[:, :, c]
        gC = g_last[:, :, c]  # (B,H,1)

        # S0^T k  ->  (B,H,C,D)
        Sk = torch.einsum("bhcd,bhde->bhce", kc, S)
        rhs = (vc - gC.unsqueeze(-1) * r_c.unsqueeze(-1) * Sk) * b_c.unsqueeze(-1)
        # U = A^-1 rhs  (A lower-triangular satuan; pakai solve, bukan inverse eksplisit)
        U = torch.linalg.solve_triangular(A_c, rhs, upper=False, unitriangular=True)

        # O = gamma * (Q S0 + tril(Q Ktilde^T, 0) U)
        qS = torch.einsum("bhcd,bhde->bhce", qc, S)
        qk = torch.matmul(qc, kt_c.transpose(-1, -2))  # (B,H,C,C)
        qk = torch.tril(qk)
        # o_t = rho_t * [ gamma_C * (Q S0) + tril(Q Ktilde^T) U ]
        o_c = r_c.unsqueeze(-1) * (gC.unsqueeze(-1) * qS + torch.matmul(qk, U))
        out[:, :, c] = o_c

        # S_end = gamma_C * S0 + Ktilde^T U
        S = gC.unsqueeze(-1) * S + torch.matmul(kt_c.transpose(-1, -2), U)

    out = out.reshape(B, H, Tp, D)[:, :, :T]
    return out


# ----------------------------------------------------------------------- modul
class GatedDeltaMemory(torch.nn.Module):
    """Proyeksi q/k/v + gate alpha/beta, lalu jalur rekuren atau chunked."""

    def __init__(self, d_model: int, n_heads: int, chunk: int = 64, bias: bool = False):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model harus habis dibagi n_heads")
        self.d, self.h = d_model, n_heads
        self.dh = d_model // n_heads
        self.chunk = chunk
        self.in_proj = torch.nn.Linear(d_model, 3 * d_model, bias=bias)
        self.gate_proj = torch.nn.Linear(d_model, 2 * n_heads, bias=True)
        self.o_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        # beta di-inisialisasi kecil: mulai sebagai "hampir tidak menulis"
        with torch.no_grad():
            self.gate_proj.bias[: n_heads].fill_(-2.0)  # alpha logits -> sigmoid ~0.12
            self.gate_proj.bias[n_heads :].fill_(-1.0)  # beta logits  -> sigmoid ~0.27

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.in_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q, k, v = (
            t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v)
        )
        q, k, v = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(
            k, dim=-1
        ), v
        g = self.gate_proj(x)  # (B,T,2H)
        alpha = torch.sigmoid(g[..., : self.h]).transpose(1, 2)
        beta = torch.sigmoid(g[..., self.h :]).transpose(1, 2)
        o = delta_chunked(q, k, v, alpha, beta, chunk=self.chunk)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(o)
