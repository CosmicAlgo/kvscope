"""
compat.py: Compatibility shims for trust_remote_code modeling files.

DeepSeek V2/V3 (and similar third-party models) ship their own
``modeling_*.py`` via ``trust_remote_code=True``. Those files were written
against older transformers versions and import symbols that newer
transformers (>=4.50) reorganized; for example
``transformers.utils.import_utils.is_torch_fx_available``, which still
exists at ``transformers.utils.is_torch_fx_available`` but no longer as a
direct attribute of the ``import_utils`` submodule.

Importing this module re-exports the moved symbols on their old paths
**before** any ``trust_remote_code`` model is loaded. Idempotent and
no-op if the running transformers already exposes the symbol.

Usage:
    # at the top of any runner that calls AutoModelForCausalLM
    # with trust_remote_code=True:
    from src.profiler import compat  # noqa: F401
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional


def _reexport(module_path: str, name: str, fallback: Optional[Callable] = None) -> None:
    """Ensure ``module_path.name`` resolves to a callable.

    If ``transformers.utils`` (or any other declared canonical location)
    exposes ``name`` but ``module_path`` no longer does, copy it across.
    If neither location has it, install ``fallback`` on ``module_path``.
    """
    import importlib

    try:
        target_mod = importlib.import_module(module_path)
    except ImportError:
        return
    if hasattr(target_mod, name):
        return  # already present, nothing to do

    # Try the new canonical home: transformers.utils
    try:
        utils_mod = importlib.import_module("transformers.utils")
    except ImportError:
        utils_mod = None

    if utils_mod is not None and hasattr(utils_mod, name):
        setattr(target_mod, name, getattr(utils_mod, name))
        return

    if fallback is not None:
        setattr(target_mod, name, fallback)
    else:
        warnings.warn(
            f"[kvscope.compat] {module_path}.{name} is missing and no fallback "
            f"supplied; trust_remote_code models that import it may fail.",
            RuntimeWarning,
        )


# ─── Concrete shims ──────────────────────────────────────────────────────────

def _detect_torch_fx() -> bool:
    try:
        import torch.fx  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_flash_attn_2() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# Apply on import so any subsequent trust_remote_code load sees the
# reconstituted module surface.
_reexport("transformers.utils.import_utils", "is_torch_fx_available",
          fallback=_detect_torch_fx)
_reexport("transformers.utils.import_utils", "is_flash_attn_2_available",
          fallback=_detect_flash_attn_2)
_reexport("transformers.utils.import_utils", "is_flash_attn_greater_or_equal_2_10",
          fallback=lambda: False)


# ─── DynamicCache legacy-attribute shims ─────────────────────────────────────
# DeepSeek-V2/V3 bundled modeling files call ``past_key_values.seen_tokens``
# and ``.get_max_length()``. Both were removed in transformers >=4.45 in
# favour of ``get_seq_length()`` and ``get_max_cache_shape()``. We restore
# the old surface as thin delegators so trust_remote_code models keep
# working without touching the cached modeling files.

def _patch_dynamic_cache() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    # seen_tokens: used by DeepSeek's prepare_inputs_for_generation
    if not hasattr(DynamicCache, "seen_tokens"):
        def _seen_tokens(self):
            # Prefer the canonical method when present
            getter = getattr(self, "get_seq_length", None)
            if callable(getter):
                try:
                    return getter()
                except Exception:
                    pass
            # Newer DynamicCache: state lives in self.layers[i].keys
            layers = getattr(self, "layers", None)
            if layers:
                k = getattr(layers[0], "keys", None)
                if k is not None and hasattr(k, "shape") and len(k.shape) >= 2:
                    return int(k.shape[-2])
            # Older DynamicCache: state lives in self.key_cache list
            kc = getattr(self, "key_cache", None)
            if isinstance(kc, list) and kc:
                k0 = kc[0]
                if hasattr(k0, "shape") and len(k0.shape) >= 2:
                    return int(k0.shape[-2])
            return 0

        DynamicCache.seen_tokens = property(_seen_tokens)

    # get_max_length: renamed to get_max_cache_shape in newer transformers
    if not hasattr(DynamicCache, "get_max_length"):
        if hasattr(DynamicCache, "get_max_cache_shape"):
            DynamicCache.get_max_length = DynamicCache.get_max_cache_shape
        else:
            DynamicCache.get_max_length = lambda self: None  # type: ignore[assignment]

    # get_usable_length(new_seq_length, layer_idx=0): used by DeepSeek's
    # forward(). In the legacy API this returned the cache length usable
    # given the about-to-arrive new tokens (handles bounded caches by
    # subtracting overflow). For unbounded DynamicCache it's just
    # get_seq_length().
    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            try:
                prev = self.get_seq_length(layer_idx)
            except TypeError:
                # newer signatures may not accept layer_idx
                prev = self.get_seq_length() if hasattr(self, "get_seq_length") else 0
            except Exception:
                prev = 0
            try:
                max_len = self.get_max_length() if hasattr(self, "get_max_length") else None
            except Exception:
                max_len = None
            if max_len is not None and prev + new_seq_length > max_len:
                return max(0, max_len - new_seq_length)
            return prev

        DynamicCache.get_usable_length = _get_usable_length

    # to_legacy_cache() / from_legacy_cache(): older DeepSeek code paths
    # occasionally round-trip through the legacy tuple format.
    if not hasattr(DynamicCache, "to_legacy_cache"):
        def _to_legacy_cache(self):
            layers = getattr(self, "layers", None)
            if layers:
                return tuple(
                    (getattr(L, "keys", None), getattr(L, "values", None))
                    for L in layers
                )
            kc = getattr(self, "key_cache", None) or []
            vc = getattr(self, "value_cache", None) or []
            return tuple(zip(kc, vc))

        DynamicCache.to_legacy_cache = _to_legacy_cache

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            inst = cls()
            if past_key_values is None:
                return inst
            for layer_idx, (k, v) in enumerate(past_key_values):
                if k is not None and v is not None:
                    try:
                        inst.update(k, v, layer_idx)
                    except Exception:
                        pass
            return inst

        DynamicCache.from_legacy_cache = _from_legacy_cache


_patch_dynamic_cache()


# ─── torch.accelerator shim (for gpt-oss-120b MXFP4 quantizer) ───────────────
# transformers >=4.48 quantizer_mxfp4.validate_environment() calls
# ``torch.accelerator.current_accelerator()``, which only exists in
# PyTorch >=2.5. On older torch (2.4.x) this raises AttributeError before
# the model ever loads. We graft a minimal shim that returns the current
# CUDA device if one is visible, else None; all the MXFP4 validator
# actually consumes.
def _patch_torch_accelerator() -> None:
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "accelerator"):
        return
    import types

    ns = types.SimpleNamespace()

    def _current_accelerator(check_available: bool = False):
        try:
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
        except Exception:
            pass
        return None

    ns.current_accelerator = _current_accelerator
    ns.is_available = lambda: torch.cuda.is_available()
    torch.accelerator = ns  # type: ignore[attr-defined]


_patch_torch_accelerator()


# ─── HfQuantizer None-config guard (for GLM-4.7-Flash) ───────────────────────
# GLM-4.7-Flash ships a ``config.json`` containing ``"quantization_config": null``.
# transformers' ``AutoHfQuantizer.supports_quant_method`` then does
# ``quantization_config_dict.get("quant_method", None)`` against ``None``,
# raising AttributeError before load even begins. Guard the classmethod so
# a null config cleanly reports "no quantization" instead of crashing.
def _patch_hf_quantizer_none_guard() -> None:
    try:
        from transformers.quantizers.auto import AutoHfQuantizer
    except ImportError:
        return

    orig = getattr(AutoHfQuantizer, "supports_quant_method", None)
    if orig is None or getattr(orig, "_kvscope_none_guarded", False):
        return

    def _safe_supports_quant_method(quantization_config_dict):
        if quantization_config_dict is None:
            return False
        return orig(quantization_config_dict)

    _safe_supports_quant_method._kvscope_none_guarded = True  # type: ignore[attr-defined]
    AutoHfQuantizer.supports_quant_method = staticmethod(_safe_supports_quant_method)


_patch_hf_quantizer_none_guard()


# ─── transformers.generation legacy class re-exports (for mamba-ssm) ─────────
# mamba-ssm 2.2.x ships ``mamba_ssm/utils/generation.py`` which does
# ``from transformers.generation import GreedySearchDecoderOnlyOutput,
# SampleDecoderOnlyOutput, TextStreamer``. The first two were removed in
# transformers >=4.50 (now unified under GenerateDecoderOnlyOutput); the
# third is still present. mamba-ssm imports this file at *top level* of
# its ``__init__.py``, so the failure blocks every Nemotron-H load even
# though Nemotron only uses the Triton kernels and never touches mamba's
# generation helpers. We graft no-op stubs onto transformers.generation
# so the eager import succeeds; the stubs are never instantiated.
def _patch_transformers_generation_legacy_exports() -> None:
    try:
        import transformers.generation as gen_mod
    except ImportError:
        return

    # Prefer the unified replacement when present; otherwise plain object.
    base = getattr(gen_mod, "GenerateDecoderOnlyOutput", None) or object

    for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if not hasattr(gen_mod, name):
            stub = type(name, (base,), {"__kvscope_stub__": True})
            setattr(gen_mod, name, stub)


_patch_transformers_generation_legacy_exports()
