"""KVSnapshot / LayerProfile accounting math."""

import pytest

from src.profiler.kv_tracer import KVSnapshot, LayerProfile


def _snap(k_bytes, v_bytes, latent_bytes=None, step=0):
    return KVSnapshot(
        step=step,
        layer_idx=0,
        layer_type="dense",
        k_shape=(1, 8, 1, 128),
        v_shape=(1, 8, 1, 128),
        k_bytes=k_bytes,
        v_bytes=v_bytes,
        k_dtype="torch.float16",
        latent_bytes=latent_bytes,
    )


def test_total_bytes_dense_is_k_plus_v():
    s = _snap(1024, 1024)
    assert s.total_bytes == 2048
    assert s.total_mb == pytest.approx(2048 / (1024 ** 2))


def test_total_bytes_mla_uses_latent_only():
    # When latent_bytes is set (MLA), only the latent is the *stored* cache.
    s = _snap(4096, 4096, latent_bytes=512)
    assert s.total_bytes == 512


def test_layer_profile_growth_and_expected_linear():
    lp = LayerProfile(layer_idx=0, layer_type="dense")
    # Three decode steps, each adding ~1 KB of K and 1 KB of V.
    for i in range(3):
        lp.snapshots.append(_snap(1024, 1024, step=i))
    curve = lp.growth_curve_mb
    assert len(curve) == 3
    assert lp.final_size_mb == curve[-1]

    expected = lp.expected_linear_mb
    per_token = lp.snapshots[0].total_mb
    assert expected == pytest.approx([per_token * (i + 1) for i in range(3)])


def test_empty_profile_is_safe():
    lp = LayerProfile(layer_idx=0, layer_type="dense")
    assert lp.growth_curve_mb == []
    assert lp.final_size_mb == 0.0
    assert lp.expected_linear_mb == []
