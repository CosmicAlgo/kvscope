"""
kv_tracer.py: KVScope Core Profiler

Instruments transformer attention layers via PyTorch forward hooks to capture
KV cache tensors at every decode step. Handles the three architecturally
distinct KV cache paradigms:

  Gemma 4:     GQA + Shared KV + Local/Global sliding-window interleaving;
               K==V constraint in global layers (unique to Gemma 4)

  GLM 5.1:     MoE + DSA (DeepSeek Sparse Attention); KV is non-deterministic
               per token due to expert routing; we capture per-expert KV load

  DeepSeek V4: MLA (Multi-Head Latent Attention); KV stored in compressed
               latent space [B, S, d_c] before uprojection to [B, H, S, d_h];
               we capture BOTH latent and expanded forms to compute compression ratio.

Usage:
    tracer = KVCacheTracer(model, model_type="gemma4")
    with tracer:
        output = model.generate(input_ids, max_new_tokens=200)
    report = tracer.report()
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pynvml
import torch
import torch.nn as nn

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class KVSnapshot:
    step: int                    # Token generation step index
    layer_idx: int               # Layer number (0-indexed)
    layer_type: str              # "local" | "global" | "dense" | "moe"
    k_shape: Tuple               # Shape of K tensor
    v_shape: Tuple               # Shape of V tensor
    k_bytes: int                 # Raw bytes of K tensor
    v_bytes: int                 # Raw bytes of V tensor
    k_dtype: str                 # e.g. "torch.float16"
    # MLA-specific (DeepSeek V4)
    latent_shape: Optional[Tuple] = None
    latent_bytes: Optional[int] = None
    compression_ratio: Optional[float] = None
    # Memory context
    gpu_alloc_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_free_mb: float = 0.0
    timestamp_ms: float = 0.0
    # GQA metadata
    n_kv_heads: int = 0
    n_q_heads: int = 0
    head_dim: int = 0

    @property
    def total_bytes(self) -> int:
        """Total KV bytes. For MLA, use latent bytes as the cached form."""
        if self.latent_bytes is not None:
            return self.latent_bytes  # Only latent is *stored* in MLA
        return self.k_bytes + self.v_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)


@dataclass
class LayerProfile:
    layer_idx: int
    layer_type: str
    snapshots: List[KVSnapshot] = field(default_factory=list)

    @property
    def growth_curve_mb(self) -> List[float]:
        return [s.total_mb for s in self.snapshots]

    @property
    def final_size_mb(self) -> float:
        return self.growth_curve_mb[-1] if self.snapshots else 0.0

    @property
    def expected_linear_mb(self) -> List[float]:
        """Expected linear growth (for leak detection baseline)."""
        if not self.snapshots:
            return []
        per_token_mb = self.snapshots[0].total_mb if self.snapshots else 0
        return [per_token_mb * (i + 1) for i in range(len(self.snapshots))]


# ─── GPU Memory Sampler (NVML-based) ─────────────────────────────────────────

class NVMLSampler:
    """Polls GPU memory via NVML. More accurate than torch.cuda.memory_allocated()
    because it sees the full driver-level allocation, including fragmentation."""

    def __init__(self, device_idx: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        self._baseline_used = 0

    def sample(self) -> Dict[str, float]:
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "total_mb": info.total / 1e6,
            "used_mb": info.used / 1e6,
            "free_mb": info.free / 1e6,
            # torch-level allocation (subset of NVML used)
            "torch_alloc_mb": torch.cuda.memory_allocated() / 1e6,
            "torch_reserved_mb": torch.cuda.memory_reserved() / 1e6,
            # Fragmentation = reserved - allocated (memory held but not used)
            "fragmentation_mb": (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1e6,
        }

    def set_baseline(self):
        """Call after model weights loaded but before generation starts."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self._baseline_used = info.used / 1e6

    def kv_overhead_mb(self) -> float:
        """VRAM used above the baseline (≈ KV cache + activations)."""
        current = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1e6
        return max(0.0, current - self._baseline_used)

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


# ─── Model-Specific Layer Classifiers ─────────────────────────────────────────

def classify_gemma4_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    Gemma 4 attention layer pattern detection.
    Gemma 4 alternates: local (sliding-window) → local → local → global (3:1 ratio).
    Local layers: SdpaAttention or GemmaAttention with window_size set.
    Global layers: attend to full context; K==V in these layers.
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None
    if "local" in name.lower() or hasattr(module, "sliding_window"):
        return "local"
    if "global" in name.lower():
        return "global"
    # Infer from layer index: Gemma 4 pattern is local/local/local/global
    parts = [p for p in name.split(".") if p.isdigit()]
    if parts:
        idx = int(parts[0])
        return "global" if (idx + 1) % 4 == 0 else "local"
    return "dense"


def classify_glm_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    GLM 5.1 / GLM-4.7 layer classifier.
    MoE layers alternate with dense layers. DSA (sparse attention) applies to
    all attention layers but with different sparsity patterns per position.
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None
    # GLM uses "num_experts" in MoE FFN layers; attention is always full
    return "moe" if hasattr(module, "num_experts") else "dense"


def classify_deepseek_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    DeepSeek V3/V4 MLA layer classifier.
    MLA is used for ALL attention layers. Key attributes:
      - module.kv_lora_rank (latent dim d_c)
      - module.q_lora_rank  (query latent dim)
      - module.qk_rope_head_dim (decoupled RoPE)
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None
    if hasattr(module, "kv_lora_rank"):
        return "mla"
    return "dense"


def _read_layer_types_from_config(module: nn.Module) -> Optional[List[str]]:
    """Best-effort lookup of ``config.layer_types`` via a model/parent reference.

    HF attention modules often expose ``self.config`` (pointing at the
    full model config). When present, we read the canonical per-layer
    attention-type list directly from the config, the only unambiguous
    ground truth for hybrid architectures like gpt-oss.
    """
    cfg = getattr(module, "config", None)
    if cfg is not None:
        lt = getattr(cfg, "layer_types", None)
        if isinstance(lt, (list, tuple)) and lt:
            return list(lt)
    return None


def classify_gptoss_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    OpenAI gpt-oss layer classifier.

    gpt-oss-120b ships an explicit ``config.layer_types`` list with one entry
    per layer, e.g. ``['sliding_attention', 'full_attention', ...]``. We read
    that directly when reachable (most robust). Fallbacks in order:

      1. ``config.layer_types[layer_idx]``   ← ground truth
      2. ``module.attention_type``           ← per-module attr if HF sets it
      3. Index-parity: layer_types[0] == 'sliding_attention' in published
         gpt-oss-120b, so even index → sliding, odd index → full.
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None

    # Extract layer index from the dotted name first (needed for ground truth)
    parts = [p for p in name.split(".") if p.isdigit()]
    layer_idx = int(parts[0]) if parts else None

    # 1. Ground truth: config.layer_types[layer_idx]
    if layer_idx is not None:
        layer_types = _read_layer_types_from_config(module)
        if layer_types and 0 <= layer_idx < len(layer_types):
            lt = layer_types[layer_idx]
            if isinstance(lt, str):
                return "sliding" if "slid" in lt.lower() else "full"

    # 2. Per-module attribute
    lt_attr = getattr(module, "attention_type", None)
    if isinstance(lt_attr, str):
        return "sliding" if "slid" in lt_attr.lower() else "full"

    # 3. Parity fallback (even → sliding, matches published gpt-oss-120b)
    if layer_idx is not None:
        return "sliding" if layer_idx % 2 == 0 else "full"
    return "dense"


def classify_mha_layer(name: str, module: nn.Module) -> Optional[str]:
    """Vanilla multi-head attention classifier, used by the MHA baseline.

    Targets pure-MHA transformer attention modules that do *not* use grouped-
    query (Q heads == KV heads), no sliding window, no MoE routing. This is
    what GPT-2, GPT-Neo, Pythia, OPT, and original LLaMA-1 use, and serves as
    the architectural anchor against which all GQA / MQA / MLA savings are
    measured in the paper.
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None
    return "mha"


def classify_lfm25_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    LiquidAI LFM2.5-350M layer classifier.

    LFM2.5 is a hybrid: 10 LIV-convolution blocks + 6 GQA attention blocks
    (16 layers total). Only the GQA blocks expose traditional K/V caches;
    the LIV conv blocks carry a fixed-size input-varying state that we do
    not hook. We therefore only classify modules whose name matches an
    attention path AND whose class name ends in ``Attention`` (the caller
    in ``KVCacheTracer.register_hooks`` enforces the class-name check).
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None
    # GQA blocks set num_key_value_heads < num_attention_heads
    n_kv = getattr(module, "num_key_value_heads", None)
    n_q = getattr(module, "num_attention_heads", None)
    if isinstance(n_kv, int) and isinstance(n_q, int) and n_kv < n_q:
        return "gqa"
    return "attn"


def classify_qwen3_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    Qwen3 dense transformer classifier.

    Qwen3-32B uses standard GQA (grouped query attention) across all layers.
    No special interleaving patterns.
    """
    if "attention" not in name.lower():
        return None
    # Qwen3 uses GQA (grouped query attention)
    return "dense"


def classify_qwen36_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    Qwen3.6 hybrid architecture classifier.

    Qwen3.6-27B has a unique hybrid architecture:
    - 48 Gated DeltaNet layers (no traditional KV cache - linear attention)
    - 16 Gated Attention layers (GQA with 4 KV heads)
    - Pattern: 16 blocks × [3 DeltaNet + 1 Attention]

    DeltaNet layers use state-space-like linear attention with minimal state.
    Only Gated Attention layers have traditional KV tensors.
    """
    # Check for attention modules (self_attn, attention, etc.)
    # Qwen3.6 uses 'self_attn' naming, not 'attention'
    has_attn = "attn" in name.lower() or "attention" in name.lower()
    if not has_attn:
        return None

    # Try to extract layer index from name (e.g., "model.layers.23.self_attn")
    layer_idx = None
    for part in name.split("."):
        if part.isdigit():
            layer_idx = int(part)
            break

    # Check module class name to determine if it's DeltaNet or Attention
    
    # DeltaNet layers use linear attention (no traditional KV cache)
    if "DeltaNet" in cls_name or "linear_attn" in name.lower():
        return "deltanet"  # Linear attention, minimal KV-like state
    
    # Attention layers use GQA with traditional KV cache
    if "Attention" in cls_name or "self_attn" in name.lower():
        return "attention"  # Traditional GQA attention with KV cache
    
    # Fallback: check layer index pattern
    # Qwen3.6 pattern: every 4th layer (indices 3, 7, 11, ...) is Gated Attention
    if layer_idx is not None:
        if layer_idx % 4 == 3:  # Layers 3, 7, 11, 15, ... (16 total)
            return "attention"
        else:
            return "deltanet"

    return None  # Unknown, skip


def classify_llama_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    Llama (Meta) classifier.

    Llama-3.x and Llama-3.1.x use grouped-query attention (GQA) for
    larger models (70B+), while smaller models (8B) use standard MHA.
    We detect GQA via num_key_value_heads < num_attention_heads and label
    accordingly. All layers are dense (no MoE).
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None
    # Direct attribute or fallback to config
    n_kv = getattr(module, "num_key_value_heads", None)
    n_q = getattr(module, "num_attention_heads", None)
    if n_kv is None or n_q is None:
        config = getattr(module, "config", None)
        if config is not None:
            if n_kv is None:
                n_kv = getattr(config, "num_key_value_heads", None)
            if n_q is None:
                n_q = getattr(config, "num_attention_heads", None)
    if isinstance(n_kv, int) and isinstance(n_q, int) and n_kv < n_q:
        return "gqa"
    return "mha"


def classify_deepseek_v4_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    DeepSeek V4-Flash layer classifier.

    V4 replaces V3's uniform MLA with a CSA/HCA hybrid:
      - CSA  (Compressed Sparse Attention): ~4x seq compression + lightning indexer
      - HCA  (Heavily Compressed Attention): ~128x seq compression, dense attend
    The published config/modeling file alternates these per layer. We read
    a per-module ``attention_type`` / ``layer_type`` attribute when the
    modeling file exposes it; otherwise fall back to kv_lora_rank (MLA
    legacy) or a parity heuristic so downstream stats still group layers.
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None

    # 1. Per-module attribute (preferred: modeling file ground truth)
    for attr in ("attention_type", "layer_type", "attn_type"):
        v = getattr(module, attr, None)
        if isinstance(v, str):
            s = v.lower()
            if "hca" in s or "heavy" in s:
                return "hca"
            if "csa" in s or "sparse" in s:
                return "csa"

    # 2. config.layer_types[idx] ground truth (same convention as gpt-oss)
    parts = [p for p in name.split(".") if p.isdigit()]
    layer_idx = int(parts[0]) if parts else None
    if layer_idx is not None:
        layer_types = _read_layer_types_from_config(module)
        if layer_types and 0 <= layer_idx < len(layer_types):
            lt = str(layer_types[layer_idx]).lower()
            if "hca" in lt or "heavy" in lt:
                return "hca"
            if "csa" in lt or "sparse" in lt:
                return "csa"

    # 3. MLA-compatibility fallback (V3-style latent KV still present)
    if hasattr(module, "kv_lora_rank"):
        return "mla"

    # 4. Parity fallback so the classifier still returns a stable label
    if layer_idx is not None:
        return "csa" if layer_idx % 2 == 0 else "hca"
    return "dense"


def classify_nemotron_layer(name: str, module: nn.Module) -> Optional[str]:
    """
    NVIDIA Nemotron-H layer classifier.

    Nemotron-3-Super-120B-A12B is NOT a pure MoE; it's a Nemotron-H hybrid
    (State-Space + Attention). The config exposes ``model_type='nemotron_h'``
    with 88 hidden layers, most of which are Mamba SSM blocks and only a
    fraction are attention. Mamba blocks maintain a constant-size SSM state
    that does not scale with sequence length, so they have no traditional KV
    cache to profile. They simply don't get hooked here (the name-based
    filter below only matches attention modules).

    Of the attention modules that DO get hooked, all are dense GQA
    (``num_key_value_heads=2``, ``num_attention_heads=32`` → 16:1 compression,
    no sliding window in the base config).
    """
    if not any(k in name for k in ["attention", "attn", "self_attn"]):
        return None
    # Optional: flag attention layers that sit next to an MoE FFN. In
    # current Nemotron-H this is rare, but the attribute probe is cheap.
    if hasattr(module, "num_experts") or hasattr(module, "num_local_experts"):
        return "moe"
    return "dense"


# ─── Main Tracer ──────────────────────────────────────────────────────────────

class KVCacheTracer:
    """
    Central profiler. Registers forward hooks on attention layers and captures
    KV tensors, GPU memory, and layer metadata at every decode step.

    Supports three model families with their distinct KV cache mechanisms:
      - "gemma4"    : GQA + Shared KV + Local/Global interleaving
      - "glm51"     : MoE + DeepSeek Sparse Attention (GLM-4.7 as proxy)
      - "deepseek"  : MLA (Multi-Head Latent Attention) compressed KV
    """

    MODEL_TYPES = {"gemma4", "glm51", "deepseek", "gptoss", "nemotron", "nemotron_h", "mha", "lfm25", "deepseek_v4", "qwen3", "qwen36", "llama", "llama4"}

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device_idx: int = 0,
        capture_every_n_steps: int = 1,  # Reduce for long sequences to save RAM
        verbose: bool = False,
    ):
        assert model_type in self.MODEL_TYPES, \
            f"model_type must be one of {self.MODEL_TYPES}, got '{model_type}'"

        self.model = model
        self.model_type = model_type
        self.verbose = verbose
        self.capture_every_n_steps = capture_every_n_steps

        self._step = 0
        self._hooks: List[Any] = []
        self._layer_profiles: Dict[int, LayerProfile] = {}
        self._raw_snapshots: List[KVSnapshot] = []
        self._active = False
        self._layers_seen_this_step: set = set()
        self._hook_debug_done = False
        # Reference to the most recently observed past_key_values cache.
        # Captured (not copied) so analyze_final_kv() can iterate over each
        # layer's K/V tensors with the Triton head-utilization kernel after
        # generation has completed.
        self._last_seen_cache: Any = None

        self.nvml = NVMLSampler(device_idx)

        # Select layer classifier
        self._classifiers = {
            "gemma4": classify_gemma4_layer,
            "glm51": classify_glm_layer,
            "deepseek": classify_deepseek_layer,
            "gptoss": classify_gptoss_layer,
            "nemotron": classify_nemotron_layer,
            "nemotron_h": classify_nemotron_layer,  # Nemotron-H uses same classifier
            "mha": classify_mha_layer,
            "lfm25": classify_lfm25_layer,
            "qwen3": classify_qwen3_layer,
            "qwen36": classify_qwen36_layer,
            "llama": classify_llama_layer,
            "llama4": classify_llama_layer,  # Llama 4 uses same attention structure as Llama 3
            "deepseek_v4": classify_deepseek_v4_layer,
        }
        self._classify = self._classifiers[model_type]

    # ── Hook Registration ─────────────────────────────────────────────────────

    def register_hooks(self):
        """Walk model and attach forward hooks to attention layers.

        Strictness rules (in order):
          1. Skip children of already-hooked attention modules (q_proj, k_proj, ...).
          2. Classifier must return a layer_type for the name.
          3. Module's class name MUST end in "Attention". This excludes inner
             helpers (e.g. ``GptOssAttentionSinkBuffer``) and weight-tied
             parameter wrappers that happen to live under an "attn" name.
          4. De-duplicate by module identity: even if the same module is
             reachable via two different names (weight-tied layers), it gets
             exactly one hook.
        """
        layer_count = 0
        hooked_prefixes: List[str] = []
        seen_module_ids: set = set()
        hooked_names: List[Tuple[str, str, str]] = []  # (name, cls_name, layer_type)

        for name, module in self.model.named_modules():
            # Debug: trace specific layers to understand why attention modules aren't hooked
            # Check layers 3, 7, 11 which should have attention
            if self.verbose:
                for check_layer in [3, 7, 11]:
                    if f".layers.{check_layer}." in name:
                        cls_name = type(module).__name__
                        print(f"[KVTracer DEBUG] Layer {check_layer} module: {name} (cls={cls_name})")
            
            # Also debug all attn/attention modules
            if self.verbose and ("attn" in name.lower() or "attention" in name.lower()):
                cls_name = type(module).__name__
                print(f"[KVTracer DEBUG] Scanning: {name} (cls={cls_name})")
            
            # 1. Skip children of already-hooked attention modules
            if any(name.startswith(p + ".") for p in hooked_prefixes):
                if self.verbose and ("attn" in name.lower() or "attention" in name.lower()):
                    print(f"[KVTracer DEBUG] Skipping (child of hooked): {name}")
                continue

            # 2. Classifier check
            layer_type = self._classify(name, module)
            if self.verbose and ("attn" in name.lower() or "attention" in name.lower()):
                print(f"[KVTracer DEBUG] Classifier returned: {layer_type} for {name}")
            if layer_type is None:
                continue

            # 3. Class-name check: must be an actual attention block.
            # Allowed: *Attention, Qwen3_5Attention, etc.
            cls_name = type(module).__name__
            is_attention_class = (
                cls_name.endswith("Attention") or
                cls_name.endswith("Attn") or
                ("attention" in cls_name.lower() and "norm" not in cls_name.lower())
            )
            if not is_attention_class:
                if self.verbose:
                    print(f"[KVTracer] Skipping non-Attention class match: "
                          f"{name} (cls={cls_name})")
                continue
            
            # Debug: Log when we find a potential attention module
            if self.verbose and ("attn" in name.lower() or "attention" in name.lower()):
                print(f"[KVTracer] Found attention candidate: {name} (cls={cls_name}, type={layer_type})")

            # 4. De-dup by module identity
            mid = id(module)
            if mid in seen_module_ids:
                if self.verbose:
                    print(f"[KVTracer] Skipping duplicate module: {name} (cls={cls_name})")
                continue
            seen_module_ids.add(mid)

            hooked_prefixes.append(name)
            hooked_names.append((name, cls_name, layer_type))

            # Determine numeric layer index from module name
            parts = [p for p in name.split(".") if p.isdigit()]
            layer_idx = int(parts[0]) if parts else layer_count

            # Initialize profile for this layer
            if layer_idx not in self._layer_profiles:
                self._layer_profiles[layer_idx] = LayerProfile(
                    layer_idx=layer_idx,
                    layer_type=layer_type,
                )

            # with_kwargs=True is required for modern HF (transformers >=4.45):
            # the attention module receives `past_key_value: Cache` as a kwarg
            # and mutates it in-place. Without this flag the hook can't see it.
            hook = module.register_forward_hook(
                self._make_hook(layer_idx, layer_type, name),
                with_kwargs=True,
            )
            self._hooks.append(hook)
            layer_count += 1

        if self.verbose:
            print(f"[KVTracer] Registered hooks on {layer_count} attention layers")
            # Print first 4 + last 1 hooked names so we can sanity-check the structure
            preview_n = min(4, len(hooked_names))
            for nm, cls, lt in hooked_names[:preview_n]:
                print(f"[KVTracer]   - {nm}  (cls={cls}, type={lt})")
            if len(hooked_names) > preview_n + 1:
                print(f"[KVTracer]   ... ({len(hooked_names) - preview_n - 1} more) ...")
            if len(hooked_names) > preview_n:
                nm, cls, lt = hooked_names[-1]
                print(f"[KVTracer]   - {nm}  (cls={cls}, type={lt})")
            # Layer-type histogram
            from collections import Counter
            type_counts = Counter(lt for _, _, lt in hooked_names)
            type_str = ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
            print(f"[KVTracer]   Layer-type histogram: {type_str}")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Hook Factory ──────────────────────────────────────────────────────────

    def _make_hook(self, layer_idx: int, layer_type: str, layer_name: str):
        def hook(module, args, kwargs, outputs):
            if not self._active:
                return

            # One-shot diagnostic: print full hook context on first fire
            if not self._hook_debug_done:
                self._hook_debug_done = True
                self._dump_hook_diagnostic(module, layer_idx, layer_name,
                                           args, kwargs, outputs)

            # Cheap per-hook bookkeeping: keep a reference to the live cache
            # so analyze_final_kv() can run Triton head-utilization on each
            # layer's K/V at end-of-generation. We capture the kwarg directly
            # (it's a Python reference, no extra memory).
            if kwargs is not None:
                # GPT-NeoX / Pythia uses the legacy kwarg name `layer_past`
                # instead of `past_key_values`. Check all three so the MHA
                # baseline runner (Pythia-1.4B) doesn't silently produce
                # empty snapshots.
                cache = (kwargs.get("past_key_values")
                         or kwargs.get("past_key_value")
                         or kwargs.get("layer_past"))
                if cache is not None:
                    self._last_seen_cache = cache

            # Auto-increment step when we revisit a layer already seen this step
            if layer_idx in self._layers_seen_this_step:
                self._step += 1
                self._layers_seen_this_step.clear()
            self._layers_seen_this_step.add(layer_idx)

            if self._step % self.capture_every_n_steps != 0:
                return

            try:
                snap = self._extract_snapshot(
                    module, layer_idx, layer_type, layer_name,
                    args=args, kwargs=kwargs, outputs=outputs,
                )
                if snap is not None:
                    self._layer_profiles[layer_idx].snapshots.append(snap)
                    self._raw_snapshots.append(snap)
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"[KVTracer] Hook error at layer {layer_idx}: {e}")

        return hook

    # ── Diagnostic Helper ─────────────────────────────────────────────────────

    def _dump_hook_diagnostic(self, module, layer_idx, layer_name,
                              args, kwargs, outputs):
        """One-shot dump of hook context to debug KV extraction."""
        print(f"\n[KVTracer DEBUG] First hook: {layer_name} (parsed layer {layer_idx})")
        print(f"[KVTracer DEBUG] Module type: {type(module).__name__}")
        print(f"[KVTracer DEBUG] module.layer_idx attr: {getattr(module, 'layer_idx', '<absent>')}")

        # args
        print(f"[KVTracer DEBUG] args len: {len(args) if args else 0}")
        for i, item in enumerate(args or ()):
            self._describe_item(f"  args[{i}]", item)

        # kwargs (modern HF puts the Cache here)
        print(f"[KVTracer DEBUG] kwargs keys: {list(kwargs.keys()) if kwargs else []}")
        for k, v in (kwargs or {}).items():
            self._describe_item(f"  kwargs[{k!r}]", v)

        # outputs
        print(f"[KVTracer DEBUG] outputs type: {type(outputs).__name__}")
        if isinstance(outputs, (tuple, list)):
            print(f"[KVTracer DEBUG] outputs len: {len(outputs)}")
            for i, item in enumerate(outputs):
                self._describe_item(f"  outputs[{i}]", item)
        else:
            self._describe_item("  outputs", outputs)
        print()

    @staticmethod
    def _describe_item(prefix, item):
        if item is None:
            print(f"[KVTracer DEBUG] {prefix}: None")
            return
        if isinstance(item, torch.Tensor):
            print(f"[KVTracer DEBUG] {prefix}: Tensor shape={tuple(item.shape)} dtype={item.dtype}")
            return
        tn = type(item).__name__
        # HF Cache objects expose key_cache / value_cache lists
        if hasattr(item, "key_cache") and hasattr(item, "value_cache"):
            kc = item.key_cache
            n = len(kc) if isinstance(kc, list) else "?"
            shape0 = kc[0].shape if isinstance(kc, list) and kc and hasattr(kc[0], "shape") else "?"
            print(f"[KVTracer DEBUG] {prefix}: {tn} (key_cache len={n}, key_cache[0].shape={shape0})")
            return
        if isinstance(item, (tuple, list)):
            print(f"[KVTracer DEBUG] {prefix}: {tn}(len={len(item)})")
            return
        # If it looks like a Cache but hasattr failed, dump introspection
        if "cache" in tn.lower() or "Cache" in tn:
            print(f"[KVTracer DEBUG] {prefix}: {tn} | MRO={[c.__name__ for c in type(item).__mro__]}")
            # Dump ALL instance attributes
            d = vars(item) if hasattr(item, '__dict__') else {}
            print(f"[KVTracer DEBUG]     __dict__ keys: {list(d.keys())}")
            for dk, dv in d.items():
                dvt = type(dv).__name__
                if isinstance(dv, (list, tuple)):
                    inner = type(dv[0]).__name__ if dv else 'empty'
                    print(f"[KVTracer DEBUG]     {dk}: {dvt}(len={len(dv)}, inner={inner})")
                    # Introspect the first layer/element if it's a cache layer
                    if dv and hasattr(dv[0], '__dict__'):
                        layer0 = dv[0]
                        l0d = vars(layer0)
                        print(f"[KVTracer DEBUG]       {inner}.__dict__ keys: {list(l0d.keys())}")
                        for lk, lv in l0d.items():
                            if isinstance(lv, torch.Tensor):
                                print(f"[KVTracer DEBUG]         {lk}: Tensor shape={tuple(lv.shape)} dtype={lv.dtype}")
                            elif isinstance(lv, (list, tuple)) and lv and hasattr(lv[0], 'shape'):
                                print(f"[KVTracer DEBUG]         {lk}: {type(lv).__name__}(len={len(lv)}, [0].shape={lv[0].shape})")
                    elif dv and hasattr(dv[0], 'shape'):
                        print(f"[KVTracer DEBUG]       [0].shape={dv[0].shape}")
                elif isinstance(dv, torch.Tensor):
                    print(f"[KVTracer DEBUG]     {dk}: Tensor shape={tuple(dv.shape)}")
                elif isinstance(dv, (int, float, bool, str, type(None))):
                    print(f"[KVTracer DEBUG]     {dk}: {dvt}={dv}")
                else:
                    print(f"[KVTracer DEBUG]     {dk}: {dvt}")
            return
        print(f"[KVTracer DEBUG] {prefix}: {tn}")

    def _extract_snapshot(
        self,
        module: nn.Module,
        layer_idx: int,
        layer_type: str,
        layer_name: str,
        args=None,
        kwargs=None,
        outputs=None,
    ) -> Optional[KVSnapshot]:
        """Extract KV tensors. Modern HF: cache lives in kwargs['past_key_value']
        and is mutated in-place during forward. Use module.layer_idx (HF's own
        canonical index) to read from cache.key_cache[layer_idx]."""
        mem = self.nvml.sample()
        ts = time.time() * 1000  # ms

        # Prefer HF's own layer_idx attr if present (more reliable than name parsing,
        # especially when vision tower attention modules are also being hooked).
        cache_layer_idx = getattr(module, "layer_idx", layer_idx)

        # ── DeepSeek MLA: compressed latent KV ────────────────────────────────
        if self.model_type == "deepseek" and layer_type == "mla":
            return self._extract_mla_snapshot(
                module, layer_idx, layer_type, mem, ts,
                args=args, kwargs=kwargs, outputs=outputs,
                cache_layer_idx=cache_layer_idx,
            )

        # ── Standard KV (Gemma 4 local/global, GLM dense) ─────────────────────
        kv_cache = self._find_kv_cache(
            module, args=args, kwargs=kwargs, outputs=outputs,
            cache_layer_idx=cache_layer_idx,
        )
        if kv_cache is None:
            return None

        k, v = kv_cache
        snap = KVSnapshot(
            step=self._step,
            layer_idx=layer_idx,
            layer_type=layer_type,
            k_shape=tuple(k.shape),
            v_shape=tuple(v.shape),
            k_bytes=k.numel() * k.element_size(),
            v_bytes=v.numel() * v.element_size(),
            k_dtype=str(k.dtype),
            gpu_alloc_mb=mem["torch_alloc_mb"],
            gpu_reserved_mb=mem["torch_reserved_mb"],
            gpu_free_mb=mem["free_mb"],
            timestamp_ms=ts,
            n_kv_heads=k.shape[1] if k.dim() == 4 else 0,
            n_q_heads=getattr(module, "num_heads", 0),
            head_dim=k.shape[-1] if k.dim() >= 2 else 0,
        )

        # Gemma 4 global layers: K == V (model ties them)
        # This means v_bytes is identical to k_bytes; flag it
        if self.model_type == "gemma4" and layer_type == "global":
            if torch.equal(k, v):
                snap.v_bytes = 0  # Don't double-count; K IS V
                snap.v_shape = ()

        return snap

    def _extract_mla_snapshot(
        self,
        module: nn.Module,
        layer_idx: int,
        layer_type: str,
        mem: Dict,
        ts: float,
        args=None,
        kwargs=None,
        outputs=None,
        cache_layer_idx: int = 0,
    ) -> Optional[KVSnapshot]:
        """
        DeepSeek MLA stores a compressed latent vector instead of full K, V.
        The latent has shape [B, S, kv_lora_rank] where kv_lora_rank << n_heads * head_dim.
        At attention compute time, it's uprojected back to [B, H, S, head_dim].
        We capture both to compute the compression ratio.
        """
        # Try common attribute names for MLA latent cache
        latent = None
        for attr in ["kv_cache", "compressed_kv", "latent_cache", "c_kv"]:
            if hasattr(module, attr):
                latent = getattr(module, attr)
                break

        # Also check past_key_value: in some MLA implementations the latent
        # is stored as a 3D tensor [B, S, d_c] vs standard 4D [B, H, S, d_h]
        kv = self._find_kv_cache(
            module, args=args, kwargs=kwargs, outputs=outputs,
            cache_layer_idx=cache_layer_idx,
        )
        k, v = (kv if kv else (None, None))

        if latent is None and k is None:
            return None

        d_c = getattr(module, "kv_lora_rank", None)
        n_heads = getattr(module, "num_heads", getattr(module, "num_key_value_heads", 0))
        head_dim = getattr(module, "head_dim", getattr(module, "v_head_dim", 0))

        # Compute compression ratio: latent_dim / (n_heads * head_dim)
        # Typical DeepSeek V3: d_c=512, n_heads=128, head_dim=128 → ratio=0.031 (97% compression)
        compression_ratio = None
        if d_c and n_heads and head_dim:
            full_dim = n_heads * head_dim
            compression_ratio = d_c / full_dim

        if latent is not None:
            lat_bytes = latent.numel() * latent.element_size()
            lat_shape = tuple(latent.shape)
        else:
            lat_bytes = None
            lat_shape = None

        k_shape = tuple(k.shape) if k is not None else ()
        v_shape = tuple(v.shape) if v is not None else ()
        k_bytes = k.numel() * k.element_size() if k is not None else 0
        v_bytes = v.numel() * v.element_size() if v is not None else 0

        return KVSnapshot(
            step=self._step,
            layer_idx=layer_idx,
            layer_type=layer_type,
            k_shape=k_shape,
            v_shape=v_shape,
            k_bytes=k_bytes,
            v_bytes=v_bytes,
            k_dtype=str(k.dtype) if k is not None else "unknown",
            latent_shape=lat_shape,
            latent_bytes=lat_bytes,
            compression_ratio=compression_ratio,
            gpu_alloc_mb=mem["torch_alloc_mb"],
            gpu_reserved_mb=mem["torch_reserved_mb"],
            gpu_free_mb=mem["free_mb"],
            timestamp_ms=ts,
            n_kv_heads=n_heads,
            n_q_heads=n_heads,
            head_dim=head_dim,
        )

    # ── KV Cache Finder ───────────────────────────────────────────────────────

    def _find_kv_cache(
        self,
        module: nn.Module,
        args=None,
        kwargs=None,
        outputs=None,
        cache_layer_idx: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Locate the per-layer K, V tensors. Modern HF (transformers >=4.45):
        the attention forward receives `past_key_value: Cache` as a kwarg
        and mutates it in-place via `cache.update(k, v, layer_idx, ...)`.
        After forward returns, `cache.key_cache[layer_idx]` is the post-update K.

        Search order:
          1. kwargs[past_key_value | past_key_values | layer_past]
                                                          (modern HF + GPT-NeoX/Pythia)
          2. positional args                             (defensive)
          3. outputs                                     (very old HF / custom impls)
          4. module attributes                           (rare)
        """
        # 1. kwargs: primary path for modern HF.
        # `layer_past` is the GPT-NeoX / Pythia legacy name; without it the
        # MHA baseline silently produces empty snapshots.
        if kwargs:
            for key in ("past_key_value", "past_key_values", "layer_past"):
                kv = self._kv_from_obj(kwargs.get(key), cache_layer_idx)
                if kv is not None:
                    return kv

        # 2. positional args: scan for any Cache-like object
        if args:
            for item in args:
                kv = self._kv_from_obj(item, cache_layer_idx)
                if kv is not None:
                    return kv

        # 3. outputs: older HF returned (attn_out, attn_weights, past_kv)
        if outputs is not None and isinstance(outputs, (tuple, list)):
            for item in outputs:
                kv = self._kv_from_obj(item, cache_layer_idx)
                if kv is not None:
                    return kv

        # 4. module attributes (legacy fallback)
        for attr in ("past_key_value", "past_key_values", "kv_cache", "cache"):
            kv = self._kv_from_obj(getattr(module, attr, None), cache_layer_idx)
            if kv is not None:
                return kv

        return None

    @staticmethod
    def _kv_from_obj(
        obj, cache_layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Try to extract (K, V) tensors from an arbitrary object.
        Handles:
          - New HF (>=4.49): cache.layers[idx] → DynamicLayer with key/value attrs
          - Old HF: cache.key_cache[idx] / cache.value_cache[idx] lists
          - HybridCache: cache.self_attention_cache sub-cache
          - Tuple-of-tensors / direct (K, V)
        Returns None if extraction fails."""
        if obj is None:
            return None

        # ── Strategy 1: New HF cache.layers[idx] → DynamicLayer ──────────────
        layers = getattr(obj, 'layers', None)
        if isinstance(layers, list) and 0 <= cache_layer_idx < len(layers):
            layer_obj = layers[cache_layer_idx]
            # Try common attribute names on the DynamicLayer
            for k_attr, v_attr in [
                ('keys', 'values'),           # New HF DynamicLayer
                ('key_cache', 'value_cache'),  # Old HF Cache
                ('key', 'value'),
                ('k', 'v'),
                ('key_states', 'value_states'),
            ]:
                k = getattr(layer_obj, k_attr, None)
                v = getattr(layer_obj, v_attr, None)
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor) and k.numel() > 0:
                    return k, v
            # If DynamicLayer is itself a tuple/list of (K, V)
            if isinstance(layer_obj, (tuple, list)) and len(layer_obj) >= 2:
                k, v = layer_obj[0], layer_obj[1]
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    return k, v

        # ── Strategy 2: Old HF cache.key_cache / cache.value_cache ───────────
        kc = getattr(obj, 'key_cache', None)
        vc = getattr(obj, 'value_cache', None)

        # HybridCache has .self_attention_cache sub-cache
        if kc is None and hasattr(obj, 'self_attention_cache'):
            sub = obj.self_attention_cache
            kc = getattr(sub, 'key_cache', None)
            vc = getattr(sub, 'value_cache', None)

        if kc is not None and vc is not None:
            if isinstance(kc, list) and isinstance(vc, list):
                if 0 <= cache_layer_idx < len(kc):
                    k, v = kc[cache_layer_idx], vc[cache_layer_idx]
                    if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor) and k.numel() > 0:
                        return k, v
                if kc and isinstance(kc[-1], torch.Tensor) and kc[-1].numel() > 0:
                    return kc[-1], vc[-1]
            elif isinstance(kc, torch.Tensor) and isinstance(vc, torch.Tensor):
                return kc, vc

        # ── Strategy 3: Direct (K, V) tuple ──────────────────────────────────
        if isinstance(obj, (tuple, list)) and len(obj) >= 2:
            k, v = obj[0], obj[1]
            if (isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)
                    and k.dim() >= 3 and v.dim() >= 3):
                return k, v

        return None

    # ── Step Counter Hook ─────────────────────────────────────────────────────

    def _on_generate_step(self):
        """Call this at the start of each generate() forward pass."""
        self._step += 1

    # ── Context Manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.register_hooks()
        self.nvml.set_baseline()
        self._active = True
        self._step = 0
        self._layers_seen_this_step.clear()
        self._hook_debug_done = False
        return self

    def __exit__(self, *args):
        self._active = False
        self.remove_hooks()

    # ── Triton head-utilization analysis ──────────────────────────────────────

    def analyze_final_kv(self) -> Dict[int, Dict[str, Any]]:
        """Run the Triton head-utilization kernel on each layer's final K/V.

        Iterates over ``self._last_seen_cache`` (captured from the hook's
        ``past_key_values`` kwarg) and computes per-head L2 norms via the
        compiled Triton kernel in :mod:`src.profiler.triton_ops`. The result
        is a per-layer dict containing ``head_utilization_ratio``,
        ``n_dead_heads`` (heads whose K-norm is < 1% of the per-layer max),
        and the mean K/V norm. Useful for surfacing whether GQA / MQA / MLA
        models are leaving KV heads effectively silent.

        Returns an empty dict if no cache has been seen yet, if Triton is not
        available, or if the cache layout is not recognised.
        """
        cache = self._last_seen_cache
        if cache is None:
            return {}

        # Triton import is deferred so the tracer module still works on
        # non-CUDA hosts (e.g. CI) where triton is not importable.
        try:
            from src.profiler.triton_ops import measure_kv_head_utilization
        except Exception as e:
            return {"_error": f"triton import failed: {e}"}

        # Gather (layer_idx, K, V) tuples from the cache. We support both the
        # legacy DynamicCache (key_cache / value_cache lists) and the modern
        # per-layer-object DynamicCache (cache.layers[i].keys/.values).
        layer_kv: List[Tuple[int, Any, Any]] = []
        if hasattr(cache, "layers") and isinstance(cache.layers, (list, tuple)):
            for li, layer in enumerate(cache.layers):
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    layer_kv.append((li, k, v))
        elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            for li, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    layer_kv.append((li, k, v))

        if not layer_kv:
            return {}

        results: Dict[int, Dict[str, Any]] = {}
        for li, k, v in layer_kv:
            try:
                with torch.no_grad():
                    stats = measure_kv_head_utilization(k.detach(), v.detach())
                results[li] = {
                    "head_utilization_ratio": float(stats["head_utilization_ratio"]),
                    "n_dead_heads": int(stats["n_dead_heads"]),
                    "n_total_heads": int(stats["n_total_heads"]),
                    "k_norm_mean": float(stats["k_norms"].mean()),
                    "v_norm_mean": float(stats["v_norms"].mean()),
                    "k_norm_std": float(stats["k_norms"].std()),
                    # Per-head K norm averaged across batch (small list)
                    "k_norms_per_head": [
                        round(x, 4) for x in stats["k_norms"].mean(0).tolist()
                    ],
                }
            except Exception as e:
                results[li] = {"error": f"{type(e).__name__}: {e}"}
        return results

    # ── Reporting ─────────────────────────────────────────────────────────────

    def report(self) -> Dict:
        """
        Generate a structured report of the profiling session.
        Suitable for JSON serialization and Prometheus export.
        """
        if not self._raw_snapshots:
            return {"error": "No snapshots captured. Did you run generation inside 'with tracer:'?"}

        total_kv_mb_final = sum(
            p.final_size_mb for p in self._layer_profiles.values()
        )
        n_local = sum(1 for p in self._layer_profiles.values() if p.layer_type == "local")
        n_global = sum(1 for p in self._layer_profiles.values() if p.layer_type == "global")
        n_mla = sum(1 for p in self._layer_profiles.values() if p.layer_type == "mla")

        # Compression stats for MLA (DeepSeek)
        mla_ratios = [
            s.compression_ratio for s in self._raw_snapshots
            if s.compression_ratio is not None
        ]

        # Per-step wall times (first timestamp seen at each step). This is
        # what we need to plot throughput at intermediate sequence-length
        # checkpoints (deliverable #8: seq length scaling) without having
        # to re-run anything. Each entry is the millisecond at which step
        # `i` started. Diff consecutive entries to get per-step latency.
        step_first_ts: Dict[int, float] = {}
        for s in self._raw_snapshots:
            if s.step not in step_first_ts or s.timestamp_ms < step_first_ts[s.step]:
                step_first_ts[s.step] = s.timestamp_ms
        if step_first_ts:
            sorted_steps = sorted(step_first_ts.keys())
            t0 = step_first_ts[sorted_steps[0]]
            step_wall_times_ms = [
                round(step_first_ts[s] - t0, 3) for s in sorted_steps
            ]
        else:
            sorted_steps = []
            step_wall_times_ms = []

        report = {
            "model_type": self.model_type,
            "total_steps": self._step,
            "total_layers_profiled": len(self._layer_profiles),
            "layer_type_counts": {
                "local": n_local,
                "global": n_global,
                "mla": n_mla,
                "dense": sum(1 for p in self._layer_profiles.values() if p.layer_type == "dense"),
                "moe": sum(1 for p in self._layer_profiles.values() if p.layer_type == "moe"),
            },
            "kv_cache": {
                "total_mb_at_end": round(total_kv_mb_final, 2),
                "per_layer_mb": {
                    idx: round(p.final_size_mb, 3)
                    for idx, p in self._layer_profiles.items()
                },
                "growth_curves": {
                    idx: [round(mb, 3) for mb in p.growth_curve_mb]
                    for idx, p in self._layer_profiles.items()
                },
            },
            "mla_compression": {
                "avg_ratio": round(float(np.mean(mla_ratios)), 4) if mla_ratios else None,
                "min_ratio": round(float(np.min(mla_ratios)), 4) if mla_ratios else None,
                "max_ratio": round(float(np.max(mla_ratios)), 4) if mla_ratios else None,
                "meaning": "Fraction of full KV dim stored in latent. 0.03 = 97% compression.",
            },
            "gpu_memory": {
                "nvml_overhead_mb": round(self.nvml.kv_overhead_mb(), 2),
                "final_sample": self.nvml.sample(),
            },
            # Wall-clock progression of decode (paper deliverable #8).
            # `step_indices` is the list of captured step indices (sorted),
            # `step_wall_times_ms` is the elapsed ms since step 0 began.
            # generate_plots.fig_throughput_vs_seqlen uses these to compute
            # tokens/sec at canonical {128, 256, 512, 1024} checkpoints.
            "decode_timing": {
                "step_indices": sorted_steps,
                "step_wall_times_ms": step_wall_times_ms,
            },
        }

        # Triton head-utilization analysis on the final K/V state of each
        # layer. This was previously dead code (`triton_ops.py` was never
        # called from the live pipeline). It now produces per-layer dead-head
        # counts that go straight into the JSON report.
        try:
            head_utilization = self.analyze_final_kv()
            if head_utilization:
                report["head_utilization"] = head_utilization
                # Aggregate (only over numeric layer entries)
                num_entries = [v for v in head_utilization.values()
                               if isinstance(v, dict) and "n_total_heads" in v]
                if num_entries:
                    total_heads = sum(e["n_total_heads"] for e in num_entries)
                    dead_heads = sum(e["n_dead_heads"] for e in num_entries)
                    avg_util = sum(e["head_utilization_ratio"]
                                   for e in num_entries) / len(num_entries)
                    report["head_utilization_summary"] = {
                        "total_heads_across_layers": int(total_heads),
                        "dead_heads_across_layers": int(dead_heads),
                        "avg_head_utilization_ratio": round(float(avg_util), 4),
                        "n_layers_analyzed": len(num_entries),
                    }
        except Exception as e:
            report["head_utilization_error"] = f"{type(e).__name__}: {e}"

        if self.verbose:
            print(f"\n[KVTracer Report]")
            print(f"  Model type       : {self.model_type}")
            print(f"  Total steps      : {self._step}")
            print(f"  Total KV at end  : {total_kv_mb_final:.1f} MB")
            print(f"  NVML KV overhead : {self.nvml.kv_overhead_mb():.1f} MB")
            if mla_ratios:
                print(f"  MLA compression  : {np.mean(mla_ratios):.3f} avg ratio")
            hu = report.get("head_utilization_summary")
            if hu:
                print(f"  Head utilization : {hu['avg_head_utilization_ratio']*100:.1f}% "
                      f"({hu['dead_heads_across_layers']}/{hu['total_heads_across_layers']} "
                      f"heads dead across {hu['n_layers_analyzed']} layers)")

        return report

    def snapshots_as_dataframe(self):
        """Export all snapshots as a pandas DataFrame for analysis.

        ``k_seq_len`` and ``v_seq_len`` are the actual cached sequence lengths
        (the second-to-last dim of K/V tensors in standard ``(B, H, S, D)``
        layout). These are the canonical token counts and should be preferred
        over the ``step`` counter for any per-token-density math, since
        ``step`` is a tracer-internal pass counter that does not always
        line up with sequence length (e.g. for sliding-window layers, S
        plateaus while step keeps incrementing).
        """
        import pandas as pd

        def _seq_len(shape):
            # Standard layout (B, H, S, D) → S is the second-to-last dim
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                return int(shape[-2])
            return 0

        records = []
        for s in self._raw_snapshots:
            records.append({
                "step": s.step,
                "layer_idx": s.layer_idx,
                "layer_type": s.layer_type,
                "total_mb": round(s.total_mb, 4),
                "k_bytes": s.k_bytes,
                "v_bytes": s.v_bytes,
                "k_seq_len": _seq_len(s.k_shape),
                "v_seq_len": _seq_len(s.v_shape),
                "latent_bytes": s.latent_bytes,
                "compression_ratio": s.compression_ratio,
                "gpu_alloc_mb": round(s.gpu_alloc_mb, 2),
                "gpu_reserved_mb": round(s.gpu_reserved_mb, 2),
                "n_kv_heads": s.n_kv_heads,
                "n_q_heads": s.n_q_heads,
                "head_dim": s.head_dim,
                "timestamp_ms": s.timestamp_ms,
            })
        return pd.DataFrame(records)
