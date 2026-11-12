#!/usr/bin/env python3
"""Test suite ASCA (tanpa pytest, cukup `python tests/run_tests.py`).

Fokus utama: **kausalitas** dan **kesesuaian implementasi cepat dengan referensi
naif**. Setiap mekanisme baru wajib lulus kedua hal ini sebelum dipakai training.

Jalankan:
    env -u PYTHONPATH ~/.venvs/ml/bin/python tests/run_tests.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from asca import ASCAConfig, ASCAModel, ablation  # noqa: E402
from asca.attention import AdaptiveHybridAttention, window_mask  # noqa: E402
from asca.experts import AdaptiveSparseMoE  # noqa: E402
from asca.memory import SegmentMemory  # noqa: E402
from asca.norm import apply_rope, rope_cache  # noqa: E402
from baseline.dense_transformer import DenseConfig, DenseTransformer  # noqa: E402

TESTS: list = []


def test(fn):
    TESTS.append(fn)
    return fn


def small_cfg(**over) -> ASCAConfig:
    kw = dict(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
        local_window=8,
        linear_chunk=8,
        mem_segment_len=16,
        n_memory_slots=4,
        n_experts=4,
        top_k=2,
        expert_ff=16,
        shared_ff=32,
        dense_ff=32,
        sem_reduction=4,
        sem_conv_kernel=4,
    )
    kw.update(over)
    return ASCAConfig(**kw)


def small_dense_cfg(**over) -> DenseConfig:
    kw = dict(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
        d_ff=32,
    )
    kw.update(over)
    return DenseConfig(**kw)


# --------------------------------------------------------------------- referensi
def naive_local(q, k, v, w):
    """Attention lokal kausal referensi (loop eksplisit)."""
    B, H, T, Dh = q.shape
    out = torch.zeros_like(q)
    for i in range(T):
        lo = max(0, i - w + 1)
        scores = (q[:, :, i].unsqueeze(-2) @ k[:, :, lo : i + 1].transpose(-1, -2)) / math.sqrt(Dh)
        probs = torch.softmax(scores, dim=-1)
        out[:, :, i] = (probs @ v[:, :, lo : i + 1]).squeeze(-2)
    return out


def naive_gated_linear(q, k, v, lam, chunk):
    """Gated linear attention referensi, bentuk rekuren per token."""
    B, H, T, Dh = q.shape
    C = chunk
    S = (T + C - 1) // C
    qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
    out = torch.zeros_like(q)
    state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype)
    ksum = torch.zeros(B, H, Dh, dtype=q.dtype)
    for s in range(S):
        t0, t1 = s * C, min(T, (s + 1) * C)
        part = torch.zeros_like(state)
        kpart = torch.zeros_like(ksum)
        for t in range(t0, t1):
            # sertakan token saat ini (j <= i) sebelum menghitung keluaran
            part = part + kf[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
            kpart = kpart + kf[:, :, t]
            num = (qf[:, :, t].unsqueeze(-2) @ (state + part)).squeeze(-2)
            den = (qf[:, :, t] * (ksum + kpart)).sum(-1, keepdim=True)
            out[:, :, t] = num / (den + 1e-6)
        state = lam[..., s].view(B, H, 1, 1) * state + part
        ksum = lam[..., s].view(B, H, 1) * ksum + kpart
    return out


# ------------------------------------------------------------------------ tests
@test
def test_rope_identity_and_norm():
    torch.manual_seed(0)
    T, Dh = 12, 8
    cos, sin = rope_cache(T, Dh, 10000.0, dtype=torch.float32)
    x = torch.randn(2, 3, T, Dh)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y[:, :, 0], x[:, :, 0], atol=1e-6), "posisi 0 harus identitas"
    n_x = x.norm(dim=-1)
    n_y = y.norm(dim=-1)
    assert torch.allclose(n_x, n_y, atol=1e-5), "RoPE harus mempertahankan norma"
    assert not torch.allclose(y, x), "RoPE harus benar-benar merotasi posisi lain"


@test
def test_window_mask_semantics():
    T, w = 8, 3
    m = window_mask(T, w, torch.device("cpu"))
    assert m.shape == (T, T)
    for i in range(T):
        for j in range(T):
            expect = (j <= i) and (j > i - w)
            assert bool(m[i, j]) == expect, f"mask[{i},{j}] salah"
    # window >= T harus setara kausal penuh
    full = window_mask(T, T, torch.device("cpu"))
    tril = torch.ones(T, T, dtype=torch.bool).tril()
    assert torch.equal(full, tril)


@test
def test_local_attention_matches_reference():
    torch.manual_seed(1)
    cfg = small_cfg()
    attn = AdaptiveHybridAttention(cfg).eval()
    B, T = 2, 16
    q = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
    k = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
    v = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
    got = attn._local(q, k, v, T)
    ref = naive_local(q, k, v, cfg.local_window)
    assert torch.allclose(got, ref, atol=1e-5), (
        f"local attention beda dari referensi (max err {(got - ref).abs().max():.3e})"
    )


@test
def test_global_linear_matches_reference():
    torch.manual_seed(2)
    for T in (16, 20):  # 20 bukan kelipatan chunk -> menguji jalur padding
        cfg = small_cfg()
        attn = AdaptiveHybridAttention(cfg).eval()
        B, C = 2, cfg.linear_chunk
        lam_val = torch.tensor([0.5, 0.8, 0.95, 0.2])
        with torch.no_grad():
            attn.decay.weight.zero_()
            attn.decay.bias.copy_(torch.log(lam_val / (1 - lam_val)))
        q = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
        k = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
        v = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
        x = torch.randn(B, T, cfg.d_model)
        got = attn._global(x, q, k, v, T)
        S = (T + C - 1) // C
        lam = lam_val.view(1, cfg.n_heads, 1).expand(B, cfg.n_heads, S).contiguous()
        ref = naive_gated_linear(q, k, v, lam, C)
        err = (got - ref).abs().max().item()
        assert torch.allclose(got, ref, atol=1e-4), f"T={T}: gated linear beda, max err {err:.3e}"


@test
def test_memory_segment_isolation():
    torch.manual_seed(3)
    cfg = small_cfg()
    mem = SegmentMemory(cfg).eval()
    B, T = 2, 32  # dua segmen (L_s = 16)
    x = torch.randn(B, T, cfg.d_model)
    with torch.no_grad():
        out = mem(x)
    assert torch.allclose(out[:, : cfg.mem_segment_len], torch.zeros_like(out[:, : cfg.mem_segment_len])), (
        "segmen 0 tidak punya masa lalu -> keluaran memori harus nol"
    )
    # ubah token di segmen 2 tidak boleh mengubah keluaran segmen 1
    x2 = x.clone()
    x2[:, cfg.mem_segment_len :] += 1.0
    with torch.no_grad():
        out2 = mem(x2)
    assert torch.allclose(
        out[:, : cfg.mem_segment_len], out2[:, : cfg.mem_segment_len], atol=1e-6
    ), "memori bocor dari segmen masa depan"


@test
def test_moe_threshold_skip_and_normalization():
    torch.manual_seed(4)
    cfg = small_cfg(router_threshold=0.0)
    moe = AdaptiveSparseMoE(cfg).eval()
    x = torch.randn(4, 8, cfg.d_model)
    with torch.no_grad():
        logits = moe.router(x.reshape(-1, cfg.d_model))
        w, _ = moe.router.route(logits)
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5), (
            "bobot top-k harus berjumlah 1"
        )
        # threshold sangat tinggi -> semua token di-skip, keluaran = shared expert saja
        moe.router.threshold = 0.999
        out_skip, _ = moe(x)
        shared_only = moe.shared(x.reshape(-1, cfg.d_model)).view_as(x)
        assert torch.allclose(out_skip, shared_only, atol=1e-6), "threshold harus mematikan routed experts"


@test
def test_moe_aux_loss_and_gradients():
    torch.manual_seed(5)
    cfg = small_cfg()
    moe = AdaptiveSparseMoE(cfg).train()
    x = torch.randn(2, 16, cfg.d_model, requires_grad=True)
    out, aux = moe(x)
    assert aux.item() > 0, "aux loss harus positif"
    (out.pow(2).mean() + aux).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in moe.experts.parameters()), (
        "tidak ada expert yang menerima gradien"
    )


@test
def test_causality_all_configs():
    """Perturbasi token terakhir tidak boleh mengubah logit posisi sebelumnya."""
    torch.manual_seed(6)
    base = small_cfg()
    configs: dict[str, torch.nn.Module] = {"A0-full": ASCAModel(base)}
    for name in ("A1", "A2", "A3", "A4", "A5", "A6"):
        configs[name] = ASCAModel(ablation(base, name))
    configs["B0-dense"] = DenseTransformer(small_dense_cfg())

    B, T = 2, 32
    x = torch.randint(0, base.vocab_size, (B, T))
    for name, model in configs.items():
        model.eval()
        with torch.no_grad():
            logits1, _ = model(x)
            x2 = x.clone()
            x2[:, -1] = (x2[:, -1] + 7) % base.vocab_size
            logits2, _ = model(x2)
        diff = (logits1[:, :-1] - logits2[:, :-1]).abs().max().item()
        assert diff < 1e-5, f"{name}: BOCOR KAUSAL, max diff {diff:.3e}"
        # dan posisi terakhir memang harus berubah (kalau tidak, modelnya mati)
        last_diff = (logits1[:, -1] - logits2[:, -1]).abs().max().item()
        assert last_diff > 1e-7, f"{name}: logit posisi terakhir tidak berubah (model tidak responsif)"


@test
def test_forward_backward_finite():
    torch.manual_seed(7)
    cfg = small_cfg()
    model = ASCAModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    total = loss + model.aux_loss
    total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} tidak menerima gradien"
        assert torch.isfinite(p.grad).all(), f"gradien {name} tidak finite"
    # loss awal harus dekat ln(V) untuk inisialisasi acak
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 1.0, (
        f"loss awal {loss.item():.3f} terlalu jauh dari ln(V)={math.log(cfg.vocab_size):.3f}"
    )


@test
def test_generate_runs():
    torch.manual_seed(8)
    cfg = small_cfg()
    model = ASCAModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(x, max_new_tokens=5, temperature=0.8, top_k=10)
    assert out.shape == (1, 13)
    assert out[:, :8].equal(x), "generate tidak boleh mengubah prompt"


@test
def test_param_accounting():
    cfg = small_cfg()
    model = ASCAModel(cfg)
    total = model.num_params()
    active = model.active_params_per_token()
    assert 0 < active < total, "active params harus lebih kecil dari total (MoE sparse)"
    dense = DenseTransformer(small_dense_cfg())
    assert dense.num_params() > 0
    print(f"      total={total:,} active={active:,} dense={dense.num_params():,}")


@test
def test_dataloader_and_tokenizer():
    from data.dataloader import TokenDataset, build_dataset
    from data.tokenizer import ByteTokenizer

    tok = ByteTokenizer()
    text = "halo dunia asca " * 50
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    ds = build_dataset(kind="synthetic", max_chars=20_000, cache=False)
    x, y = ds.get_batch("train", batch_size=2, seq_len=16)
    assert x.shape == (2, 16) and y.shape == (2, 16)
    assert torch.equal(y[:, :-1], x[:, 1:]), "y harus x yang digeser satu posisi"
    assert ds.n_tokens("val") > 0
    del TokenDataset


def main() -> int:
    torch.set_grad_enabled(True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    failed = []
    print(f"Menjalankan {len(TESTS)} test (torch {torch.__version__})\n")
    for fn in TESTS:
        if only and only not in fn.__name__:
            continue
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append((fn.__name__, exc))
            print(f"  FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
    print()
    if failed:
        print(f"{len(failed)} test GAGAL: {[f[0] for f in failed]}")
        return 1
    print("Semua test lulus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
