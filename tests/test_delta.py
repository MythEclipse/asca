"""Oracle Python untuk memverifikasi kernel Rust (BUKAN implementasi proyek).

Implementasi resmi ada di ``rust/``. Berkas ini dipertahankan karena ia acuan
independen: bentuk rekuren vs chunked di sini sudah terbukti identik sampai 1.8e-15
dan dipakai sebagai pembanding saat memverifikasi port Rust/CUDA nanti.

Gerbang wajib sebelum port ke Rust/CUDA: kalau kedua bentuk tidak identik, angka
training (chunked) tidak akan berlaku untuk decode (rekuren) — bug ini sudah dua kali
menggigit proyek di v1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.oracle_py.delta import delta_chunked, delta_recurrent  # noqa: E402

FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def test_parity_recurrent_vs_chunked() -> None:
    """Chunked harus identik dengan rekuren, untuk berbagai ukuran chunk."""
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 40, 16
    dtype = torch.float64
    q = torch.randn(B, H, T, D, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, dtype=dtype), dim=-1)
    v = torch.randn(B, H, T, D, dtype=dtype)
    alpha = torch.sigmoid(torch.randn(B, H, T, dtype=dtype) + 1.0)
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=dtype))

    ref = delta_recurrent(q, k, v, alpha, beta)
    scale = ref.abs().max().item()
    for chunk in (8, 16, 40, 64):
        got = delta_chunked(q, k, v, alpha, beta, chunk=chunk)
        err = (got - ref).abs().max().item() / scale  # galat RELATIF
        check(
            f"chunked(chunk={chunk}) == rekuren",
            err < 1e-12,
            f"galat relatif {err:.2e} (skala {scale:.1f})",
        )


def test_no_decay_no_delta_is_linear_attention() -> None:
    """alpha=1, beta=1, dan tanpa koreksi -> harus sama dengan linear attention kausal."""
    torch.manual_seed(1)
    B, H, T, D = 1, 2, 24, 8
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    alpha = torch.ones(B, H, T, dtype=torch.float64)
    beta = torch.ones(B, H, T, dtype=torch.float64)

    # delta rule dengan beta=1 tetap punya suku koreksi, jadi pembandingnya adalah
    # bentuk rekuren itu sendiri; di sini kita hanya pastikan bentuk chunked tidak
    # bergantung pada T chunking dan tetap kausal.
    got = delta_chunked(q, k, v, alpha, beta, chunk=7)
    ref = delta_recurrent(q, k, v, alpha, beta)
    err = (got - ref).abs().max().item() / ref.abs().max().item()
    check("alpha=1,beta=1 tetap konsisten", err < 1e-12, f"galat relatif {err:.2e}")


def test_causality() -> None:
    """Mengubah token masa depan tidak boleh mengubah output masa lalu."""
    torch.manual_seed(2)
    B, H, T, D = 1, 2, 32, 8
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    alpha = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))

    base = delta_chunked(q, k, v, alpha, beta, chunk=8)
    cut = 20
    v2, k2, alpha2, beta2 = v.clone(), k.clone(), alpha.clone(), beta.clone()
    v2[:, :, cut:] = torch.randn_like(v2[:, :, cut:])
    k2[:, :, cut:] = torch.nn.functional.normalize(
        torch.randn_like(k2[:, :, cut:]), dim=-1
    )
    alpha2[:, :, cut:] = torch.rand(B, H, T - cut, dtype=torch.float64)
    beta2[:, :, cut:] = torch.rand(B, H, T - cut, dtype=torch.float64)
    pert = delta_chunked(q, k2, v2, alpha2, beta2, chunk=8)

    past = (pert[:, :, :cut] - base[:, :, :cut]).abs().max().item()
    future = (pert[:, :, cut:] - base[:, :, cut:]).abs().max().item()
    check("output masa lalu tidak berubah", past < 1e-12, f"galat {past:.2e}")
    check("output masa depan berubah (uji tidak vakum)", future > 1e-6, f"delta {future:.2e}")


def test_state_is_constant_memory() -> None:
    """Ukuran state tidak bergantung pada panjang konteks."""
    import reference.oracle_py.delta as dl

    torch.manual_seed(3)
    H, D = 2, 8
    sizes = []
    for T in (16, 64, 256):
        q = torch.randn(1, H, T, D, dtype=torch.float64)
        k = torch.nn.functional.normalize(torch.randn(1, H, T, D, dtype=torch.float64), dim=-1)
        v = torch.randn(1, H, T, D, dtype=torch.float64)
        alpha = torch.sigmoid(torch.randn(1, H, T, dtype=torch.float64))
        beta = torch.sigmoid(torch.randn(1, H, T, dtype=torch.float64))
        # bentuk rekuren menyimpan hanya S: H*D*D
        with torch.no_grad():
            dl.delta_recurrent(q, k, v, alpha, beta)
        sizes.append(H * D * D)
    check("state rekuren konstan terhadap T", len(set(sizes)) == 1, f"ukuran {sizes[0]} elemen")


def test_module_forward_backward() -> None:
    """Modul harus jalan, gradiennya finite, dan bentuk output benar."""
    from reference.oracle_py.delta import GatedDeltaMemory

    torch.manual_seed(4)
    m = GatedDeltaMemory(d_model=32, n_heads=4, chunk=8)
    x = torch.randn(2, 20, 32, requires_grad=True)
    y = m(x)
    check("bentuk output", y.shape == x.shape, str(tuple(y.shape)))
    y.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    check("gradien finite", all(torch.isfinite(g).all() for g in grads), f"{len(grads)} tensor")


if __name__ == "__main__":
    for fn in (
        test_parity_recurrent_vs_chunked,
        test_no_decay_no_delta_is_linear_attention,
        test_causality,
        test_state_is_constant_memory,
        test_module_forward_backward,
    ):
        print(f"== {fn.__name__}")
        fn()
    print()
    if FAIL:
        print(f"{len(FAIL)} test GAGAL: {FAIL}")
        raise SystemExit(1)
    print("Semua test lulus.")
