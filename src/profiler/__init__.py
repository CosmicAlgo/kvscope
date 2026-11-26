"""KVScope profiler: core instrumentation, leak detection, and GPU analysis."""

from src.profiler.kv_tracer import (
    KVCacheTracer,
    KVSnapshot,
    LayerProfile,
    NVMLSampler,
    LAYER_CLASSIFIERS,
)
from src.profiler.leak_detector import KVLeakDetector, DetectorReport, LeakFinding
from src.profiler.adapters import (
    ArchitectureAdapter,
    ADAPTERS,
    get_adapter,
    analytical_density_kb_per_token_per_layer,
)

__all__ = [
    "KVCacheTracer",
    "KVSnapshot",
    "LayerProfile",
    "NVMLSampler",
    "LAYER_CLASSIFIERS",
    "KVLeakDetector",
    "DetectorReport",
    "LeakFinding",
    "ArchitectureAdapter",
    "ADAPTERS",
    "get_adapter",
    "analytical_density_kb_per_token_per_layer",
]
