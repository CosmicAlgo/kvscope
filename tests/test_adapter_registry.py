"""Adapter registry consistency + analytical density model.

These tests need only torch/numpy/pynvml (no CUDA, no model weights), so they
run anywhere including CI.
"""

import pytest

from src.profiler.adapters import (
    ADAPTERS,
    analytical_density_kb_per_token_per_layer,
    get_adapter,
)
from src.profiler.kv_tracer import LAYER_CLASSIFIERS, KVCacheTracer


def test_registry_in_sync_with_classifiers():
    assert set(ADAPTERS) == set(LAYER_CLASSIFIERS)
    assert set(ADAPTERS) == set(KVCacheTracer.MODEL_TYPES)


def test_every_adapter_has_callable_classifier():
    for mt, adapter in ADAPTERS.items():
        assert adapter.classifier is not None, f"{mt} missing classifier binding"
        assert callable(adapter.classifier)


def test_minimax_m3_registered_with_pending_weights():
    a = get_adapter("minimax_m3")
    assert a.kv_strategy == "msa_block_sparse"
    assert a.weights_public is False
    assert a.kv_compressed is False  # MSA stores full-precision K/V
    assert a.supports_long_context is True
    assert a.max_context == 1_000_000
    # While weights are pending, the testable model must be the M2 stand-in.
    assert a.testable_model == "MiniMaxAI/MiniMax-M2"


def test_get_adapter_unknown_raises():
    with pytest.raises(KeyError):
        get_adapter("definitely_not_a_model")


def test_analytical_density_dense_gqa():
    # 8 KV heads * 128 head_dim * 2 (K and V) * 2 bytes (bf16) / 1024 = 4.0 KB
    kb = analytical_density_kb_per_token_per_layer(
        "gemma4", num_key_value_heads=8, head_dim=128
    )
    assert kb == pytest.approx(4.0)


def test_analytical_density_msa_matches_dense_storage():
    # MSA stores uncompressed K/V, so per-token storage equals GQA.
    kb = analytical_density_kb_per_token_per_layer(
        "minimax_m3", num_key_value_heads=8, head_dim=128
    )
    assert kb == pytest.approx(4.0)


def test_analytical_density_mla_uses_latent_only():
    # MLA stores only the latent of width kv_lora_rank (no factor of 2).
    kb = analytical_density_kb_per_token_per_layer("deepseek", kv_lora_rank=512)
    assert kb == pytest.approx(1.0)


def test_analytical_density_returns_none_when_underspecified():
    assert analytical_density_kb_per_token_per_layer("gemma4") is None
    assert analytical_density_kb_per_token_per_layer("deepseek") is None
