"""Triton-kernel correctness via the pure-PyTorch fallback path.

``triton_ops`` imports ``triton`` at module load, so the whole module is skipped
when triton is unavailable (it is an optional extra). The CPU/torch fallback is
exercised everywhere; the Triton-vs-torch parity check only runs when CUDA is
present.
"""

import pytest

triton = pytest.importorskip("triton")  # noqa: F841  (skip file if not installed)

import torch  # noqa: E402

from src.profiler.triton_ops import (  # noqa: E402
    compute_attention_entropy,
    measure_kv_head_utilization,
)


def test_head_utilization_torch_backend_detects_dead_head():
    torch.manual_seed(0)
    k = torch.randn(1, 4, 16, 8)
    v = torch.randn(1, 4, 16, 8)
    k[:, 0] = 0.0  # zero out head 0 -> "dead"
    v[:, 0] = 0.0

    out = measure_kv_head_utilization(k, v, backend="torch")
    assert out["n_total_heads"] == 4
    assert out["n_dead_heads"] == 1
    assert out["head_utilization_ratio"] == pytest.approx(0.75)
    assert bool(out["dead_head_mask"][0]) is True


def test_attention_entropy_ranges():
    torch.manual_seed(0)
    attn = torch.softmax(torch.randn(1, 4, 16, 16), dim=-1)
    out = compute_attention_entropy(attn)
    assert 0.0 <= out["mean_entropy"] <= out["max_entropy_possible"] + 1e-6
    assert 0.0 <= out["low_entropy_head_fraction"] <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for Triton")
def test_triton_matches_torch_head_norms():
    torch.manual_seed(0)
    k = torch.randn(1, 8, 64, 16, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 8, 64, 16, device="cuda", dtype=torch.float16)
    triton_out = measure_kv_head_utilization(k, v, backend="triton")
    torch_out = measure_kv_head_utilization(k, v, backend="torch")
    max_diff = float((triton_out["k_norms"] - torch_out["k_norms"]).abs().max())
    assert max_diff < 1e-2
