"""Layer-classifier unit tests.

The Qwen3.6 test is a regression guard for a real bug: ``classify_qwen36_layer``
referenced an undefined ``cls_name`` and raised ``NameError`` on every attention
module before the fix.
"""

from types import SimpleNamespace

from src.profiler.kv_tracer import (
    classify_deepseek_layer,
    classify_deepseek_v4_layer,
    classify_gemma4_layer,
    classify_gptoss_layer,
    classify_mha_layer,
    classify_minimax_m3_layer,
    classify_qwen36_layer,
)


def _named_module(cls_name):
    """Build a throwaway module instance whose class name is ``cls_name``."""
    return type(cls_name, (), {})()


def _with_attrs(cls_name, attrs):
    obj = type(cls_name, (), {})()
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def test_mha_classifier():
    assert classify_mha_layer("model.layers.0.self_attn", object()) == "mha"
    # Non-attention names are ignored.
    assert classify_mha_layer("model.layers.0.mlp", object()) is None


def test_gemma4_local_global_by_index():
    # idx 3 -> (3+1) % 4 == 0 -> global; idx 0 -> local
    assert classify_gemma4_layer("model.layers.3.self_attn", SimpleNamespace()) == "global"
    assert classify_gemma4_layer("model.layers.0.self_attn", SimpleNamespace()) == "local"


def test_qwen36_regression_no_nameerror():
    # Previously raised NameError (cls_name undefined). Now classifies cleanly.
    deltanet = _named_module("Qwen3DeltaNet")
    attn = _named_module("Qwen3Attention")
    assert classify_qwen36_layer("model.layers.0.linear_attn", deltanet) == "deltanet"
    assert classify_qwen36_layer("model.layers.3.self_attn", attn) == "attention"


def test_deepseek_mla_via_attr():
    mod = _with_attrs("DeepseekAttention", {"kv_lora_rank": 512})
    assert classify_deepseek_layer("model.layers.0.self_attn", mod) == "mla"


def test_deepseek_v4_csa_hca_via_attr():
    hca = _with_attrs("DSV4Attention", {"attention_type": "HCA"})
    csa = _with_attrs("DSV4Attention", {"attention_type": "csa_sparse"})
    assert classify_deepseek_v4_layer("model.layers.0.self_attn", hca) == "hca"
    assert classify_deepseek_v4_layer("model.layers.1.self_attn", csa) == "csa"


def test_gptoss_reads_config_layer_types():
    cfg = SimpleNamespace(layer_types=["sliding_attention", "full_attention"])
    mod = SimpleNamespace(config=cfg)
    assert classify_gptoss_layer("model.layers.0.self_attn", mod) == "sliding"
    assert classify_gptoss_layer("model.layers.1.self_attn", mod) == "full"


def test_minimax_m3_defaults_to_msa():
    # No hints -> MSA (M3 applies MSA on every layer).
    assert classify_minimax_m3_layer("model.layers.0.self_attn", SimpleNamespace()) == "msa"


def test_minimax_m3_respects_layer_types_and_attrs():
    cfg = SimpleNamespace(layer_types=["full_attention", "msa"])
    mod = SimpleNamespace(config=cfg)
    assert classify_minimax_m3_layer("model.layers.0.self_attn", mod) == "full"
    assert classify_minimax_m3_layer("model.layers.1.self_attn", mod) == "msa"

    attr_mod = _with_attrs("MiniMaxM3Attention", {"attention_type": "full"})
    assert classify_minimax_m3_layer("model.layers.0.self_attn", attr_mod) == "full"


def test_minimax_m3_detects_block_attrs():
    mod = _with_attrs("MiniMaxM3Attention", {"msa_block_size": 128})
    assert classify_minimax_m3_layer("model.layers.0.self_attn", mod) == "msa"


def test_non_attention_names_ignored_across_classifiers():
    for fn in (classify_gemma4_layer, classify_deepseek_layer,
               classify_minimax_m3_layer, classify_gptoss_layer):
        assert fn("model.layers.0.mlp.up_proj", SimpleNamespace()) is None
