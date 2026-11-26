"""
KVScope API behavioural profiler.

KVScope's in-process tracer cannot run against a hosted endpoint (no tensors,
no NVML). This package is the *complement*: a black-box profiler that infers an
architecture's inference-time signature from timing and output quality alone.

It was written for models that are reachable only through an API (MiniMax M3
was the first target) and measures:

  * prefill scaling   — TTFT vs input length
  * decode scaling    — inter-token latency vs context length
  * effective context — needle-in-a-haystack recall across depth x length

The vendor's speedup figures for MiniMax Sparse Attention (~9.7x prefill,
~15.6x decode at 1M tokens) are claims this tool can test; no results from it
are in this repository yet.

Entry point: ``kvscope api ...`` (see ``src/api/cli.py``).
"""

from src.api.client import ChatClient, PROVIDERS
from src.api.metrics import LatencyTrace, fit_power_law, summarize_traces

__all__ = [
    "ChatClient",
    "PROVIDERS",
    "LatencyTrace",
    "fit_power_law",
    "summarize_traces",
]
