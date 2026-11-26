"""
adapters.py: KVScope Architecture Adapter Registry

A declarative, single-source-of-truth description of every architecture KVScope
knows how to profile. Each :class:`ArchitectureAdapter` bundles:

  * the ``model_type`` key (matches ``KVCacheTracer.MODEL_TYPES``),
  * the layer classifier used to attach hooks (re-used from ``kv_tracer``),
  * the KV-cache *strategy* and whether stored KV is compressed,
  * default / stand-in HuggingFace model ids,
  * an *analytical* KB-per-token-per-layer model, so empirical measurements can
    be checked against theory (see ``scripts/analytical_validation.py``).

Why a registry?  Before this, architecture knowledge was scattered across a
1,100-line bash file, a hard-coded ``MODEL_TYPES`` set, and an ``if/elif`` ladder
in each runner. Adding MiniMax M3 (MSA) now means: one classifier in
``kv_tracer.py`` + one entry here. A unit test
(``tests/test_adapter_registry.py``) asserts the two stay in sync.

This module imports from ``kv_tracer`` (one-directional); ``kv_tracer`` never
imports this module, so there is no circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from src.profiler.kv_tracer import LAYER_CLASSIFIERS

# ─── KV strategy vocabulary ───────────────────────────────────────────────────
# Short, stable codes used in reports / paper tables.

MHA = "mha"                       # pure multi-head attention (num_kv == num_q)
GQA = "gqa"                       # grouped-query attention
GQA_LOCAL_GLOBAL = "gqa_local_global"   # Gemma-style sliding/global interleave
GQA_SLIDING_FULL = "gqa_sliding_full"   # gpt-oss alternating sliding/full
MLA = "mla"                       # multi-head latent attention (compressed KV)
CSA_HCA = "csa_hca"               # DeepSeek V4 compressed/heavily-compressed
MSA_BLOCK_SPARSE = "msa_block_sparse"   # MiniMax Sparse Attention
SSM_GQA = "ssm_gqa"               # Mamba SSM majority + sparse GQA (Nemotron-H)
LIV_GQA = "liv_gqa"               # LFM2.5 LIV convolution + GQA
DELTANET_GQA = "deltanet_gqa"     # Qwen3.6 Gated DeltaNet + GQA


@dataclass(frozen=True)
class ArchitectureAdapter:
    """Declarative description of one profilable architecture."""

    model_type: str
    display_name: str
    kv_strategy: str
    # Is the *stored* KV compressed relative to dense [B, H, S, D]?  MLA/CSA: yes.
    # MSA keeps K/V uncompressed (sparsity is in the read pattern, not storage).
    kv_compressed: bool = False
    # Primary HF id (may be gated / unreleased) then any drop-in stand-ins used
    # for code-path testing before the real weights are public.
    hf_models: Tuple[str, ...] = ()
    standin_models: Tuple[str, ...] = ()
    supports_long_context: bool = False
    max_context: Optional[int] = None
    weights_public: bool = True
    notes: str = ""
    # Bound after the class is defined, from LAYER_CLASSIFIERS.
    classifier: Optional[Callable] = field(default=None, compare=False, repr=False)

    @property
    def primary_model(self) -> Optional[str]:
        return self.hf_models[0] if self.hf_models else None

    @property
    def testable_model(self) -> Optional[str]:
        """Best model to exercise the code path *today* (stand-in if weights
        are not yet public, otherwise the primary)."""
        if not self.weights_public and self.standin_models:
            return self.standin_models[0]
        return self.primary_model or (self.standin_models[0] if self.standin_models else None)


# ─── Registry ─────────────────────────────────────────────────────────────────

_ADAPTERS_LIST = [
    ArchitectureAdapter(
        model_type="mha",
        display_name="Pythia-1.4B (pure MHA anchor)",
        kv_strategy=MHA,
        hf_models=("EleutherAI/pythia-1.4b-deduped",),
        notes="Architectural baseline: num_kv_heads == num_attention_heads, no compression.",
    ),
    ArchitectureAdapter(
        model_type="gemma4",
        display_name="Gemma 4",
        kv_strategy=GQA_LOCAL_GLOBAL,
        hf_models=("google/gemma-4-9b-it",),
        supports_long_context=True,
        notes="GQA + shared KV + local/global interleave; K==V in global layers.",
    ),
    ArchitectureAdapter(
        model_type="glm51",
        display_name="GLM-4.7-Flash (DSA proxy)",
        kv_strategy=GQA,
        hf_models=("zai-org/GLM-4.7-Flash",),
        notes="MoE + DeepSeek Sparse Attention; per-token KV non-deterministic via routing.",
    ),
    ArchitectureAdapter(
        model_type="gptoss",
        display_name="gpt-oss-120B",
        kv_strategy=GQA_SLIDING_FULL,
        hf_models=("openai/gpt-oss-120b",),
        notes="Alternating sliding/full attention + MoE FFN; strong layer bimodality.",
    ),
    ArchitectureAdapter(
        model_type="deepseek",
        display_name="DeepSeek (MLA)",
        kv_strategy=MLA,
        kv_compressed=True,
        hf_models=("deepseek-ai/DeepSeek-V2-Lite",),
        notes="Multi-head latent attention; only the latent [B,S,d_c] is stored.",
    ),
    ArchitectureAdapter(
        model_type="deepseek_v4",
        display_name="DeepSeek V4-Flash (CSA/HCA)",
        kv_strategy=CSA_HCA,
        kv_compressed=True,
        hf_models=("deepseek-ai/DeepSeek-V4-Flash",),
        supports_long_context=True,
        max_context=1_000_000,
        notes="4x/128x sequence compression; ~2% KV vs traditional GQA.",
    ),
    ArchitectureAdapter(
        model_type="nemotron",
        display_name="Nemotron-H (SSM + GQA)",
        kv_strategy=SSM_GQA,
        notes="Mamba SSM majority + sparse GQA attention; SSM blocks carry no KV cache.",
    ),
    ArchitectureAdapter(
        model_type="nemotron_h",
        display_name="Nemotron-H (SSM + GQA)",
        kv_strategy=SSM_GQA,
        notes="Alias of 'nemotron' (same classifier).",
    ),
    ArchitectureAdapter(
        model_type="lfm25",
        display_name="LFM2.5-350M (LIV + GQA)",
        kv_strategy=LIV_GQA,
        hf_models=("LiquidAI/LFM2.5-350M",),
        notes="10 LIV-convolution blocks + 6 GQA blocks; only GQA blocks expose KV.",
    ),
    ArchitectureAdapter(
        model_type="qwen3",
        display_name="Qwen3 (dense GQA)",
        kv_strategy=GQA,
        hf_models=("Qwen/Qwen3-32B",),
    ),
    ArchitectureAdapter(
        model_type="qwen36",
        display_name="Qwen3.6 (DeltaNet + GQA)",
        kv_strategy=DELTANET_GQA,
        notes="48 Gated DeltaNet + 16 Gated Attention layers; DeltaNet carries no KV cache.",
    ),
    ArchitectureAdapter(
        model_type="llama",
        display_name="Llama 3.x",
        kv_strategy=GQA,
        hf_models=("meta-llama/Meta-Llama-3.1-8B-Instruct",),
    ),
    ArchitectureAdapter(
        model_type="llama4",
        display_name="Llama 4",
        kv_strategy=GQA,
        notes="Same attention structure as Llama 3 for hooking purposes.",
    ),
    ArchitectureAdapter(
        model_type="minimax_m3",
        display_name="MiniMax M3 (MSA)",
        kv_strategy=MSA_BLOCK_SPARSE,
        kv_compressed=False,  # MSA keeps K/V uncompressed; sparsity is in the read pattern
        hf_models=("MiniMaxAI/MiniMax-M3",),
        standin_models=("MiniMaxAI/MiniMax-M2",),
        supports_long_context=True,
        max_context=1_000_000,
        weights_public=False,  # as of May 2026; not re-checked since
        notes=(
            "MiniMax Sparse Attention: partitions KV into blocks read once "
            "('KV outer gather Q'), full-precision K/V, 1M context, native "
            "multimodal. Weights were not public in May 2026; use the M2 stand-in "
            "for code-path tests."
        ),
    ),
]

ADAPTERS: Dict[str, ArchitectureAdapter] = {a.model_type: a for a in _ADAPTERS_LIST}

# Bind each adapter's classifier from the tracer registry, and fail loudly at
# import time if the two ever drift out of sync. ``frozen=True`` dataclasses use
# object.__setattr__ to set the post-init field.
_missing_adapter = set(LAYER_CLASSIFIERS) - set(ADAPTERS)
_missing_classifier = set(ADAPTERS) - set(LAYER_CLASSIFIERS)
if _missing_adapter or _missing_classifier:
    raise RuntimeError(
        "adapters.ADAPTERS and kv_tracer.LAYER_CLASSIFIERS are out of sync. "
        f"Missing adapters for: {sorted(_missing_adapter)}; "
        f"missing classifiers for: {sorted(_missing_classifier)}."
    )
for _mt, _adapter in ADAPTERS.items():
    object.__setattr__(_adapter, "classifier", LAYER_CLASSIFIERS[_mt])


def get_adapter(model_type: str) -> ArchitectureAdapter:
    if model_type not in ADAPTERS:
        raise KeyError(
            f"Unknown model_type '{model_type}'. Known: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[model_type]


# ─── Analytical KV density model ──────────────────────────────────────────────

def analytical_density_kb_per_token_per_layer(
    model_type: str,
    *,
    num_key_value_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    dtype_bytes: int = 2,
) -> Optional[float]:
    """Theoretical KB of *stored* KV per token per attention layer.

    Returns ``None`` when the formula needs config values that were not
    supplied (caller should then fall back to empirical measurement only).

    * Dense families (MHA/GQA/local-global/sliding-full/MSA): K and V are both
      stored uncompressed → ``num_kv_heads * head_dim * 2 * dtype_bytes``.
      MSA is included here deliberately: its speedup comes from the *read*
      pattern, not from shrinking stored KV, so per-token storage matches GQA.
    * MLA / CSA-HCA: only the latent of width ``kv_lora_rank`` is stored →
      ``kv_lora_rank * dtype_bytes`` (no factor of 2; K and V share the latent).
    """
    adapter = get_adapter(model_type)
    strategy = adapter.kv_strategy

    if strategy in (MLA, CSA_HCA):
        if kv_lora_rank is None:
            return None
        return (kv_lora_rank * dtype_bytes) / 1024.0

    # Dense storage families (MSA stores full-precision K/V like GQA).
    if num_key_value_heads is None or head_dim is None:
        return None
    return (num_key_value_heads * head_dim * 2 * dtype_bytes) / 1024.0


__all__ = [
    "ArchitectureAdapter",
    "ADAPTERS",
    "get_adapter",
    "analytical_density_kb_per_token_per_layer",
    "MHA", "GQA", "GQA_LOCAL_GLOBAL", "GQA_SLIDING_FULL", "MLA", "CSA_HCA",
    "MSA_BLOCK_SPARSE", "SSM_GQA", "LIV_GQA", "DELTANET_GQA",
]
