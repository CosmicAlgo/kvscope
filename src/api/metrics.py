"""
src/api/metrics.py: timing math for the API behavioural profiler.

A :class:`LatencyTrace` records the timestamps of a single streamed generation
and derives the standard inference-serving metrics (TTFT, ITL, throughput).
``fit_power_law`` recovers the scaling exponent of a metric vs context length —
the headline number for validating MSA's sub-quadratic claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LatencyTrace:
    """Timing record for one streamed generation."""

    input_tokens: int
    t_send: float                       # perf_counter() right before request
    token_timestamps: List[float] = field(default_factory=list)
    t_end: Optional[float] = None
    output_tokens: int = 0
    label: str = ""

    # ── derived metrics ─────────────────────────────────────────────────────────

    @property
    def ttft_s(self) -> Optional[float]:
        """Time to first token (prefill-dominated)."""
        if not self.token_timestamps:
            return None
        return self.token_timestamps[0] - self.t_send

    @property
    def inter_token_latencies_s(self) -> List[float]:
        """Gaps between successive streamed tokens (decode-dominated)."""
        ts = self.token_timestamps
        return [ts[i] - ts[i - 1] for i in range(1, len(ts))]

    @property
    def mean_itl_s(self) -> Optional[float]:
        itls = self.inter_token_latencies_s
        return float(np.mean(itls)) if itls else None

    def itl_percentile(self, p: float) -> Optional[float]:
        itls = self.inter_token_latencies_s
        return float(np.percentile(itls, p)) if itls else None

    @property
    def decode_tokens_per_s(self) -> Optional[float]:
        """Output tokens per second over the decode phase (excludes prefill)."""
        ts = self.token_timestamps
        if len(ts) < 2:
            return None
        span = ts[-1] - ts[0]
        return (len(ts) - 1) / span if span > 0 else None

    @property
    def total_latency_s(self) -> Optional[float]:
        if self.t_end is None:
            return None
        return self.t_end - self.t_send

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens or len(self.token_timestamps),
            "ttft_s": _round(self.ttft_s),
            "mean_itl_s": _round(self.mean_itl_s),
            "p50_itl_s": _round(self.itl_percentile(50)),
            "p95_itl_s": _round(self.itl_percentile(95)),
            "decode_tokens_per_s": _round(self.decode_tokens_per_s),
            "total_latency_s": _round(self.total_latency_s),
        }


def _round(x: Optional[float], n: int = 5) -> Optional[float]:
    return round(x, n) if isinstance(x, (int, float)) else None


def fit_power_law(x: List[float], y: List[float]) -> Dict[str, Optional[float]]:
    """Fit ``y = a * x**b`` via least squares in log-log space.

    The exponent ``b`` is the interesting quantity:
      * b ~ 0  → flat (per-token cost independent of context; ideal sparse decode)
      * b ~ 1  → linear
      * b ~ 2  → quadratic (dense full-attention prefill)

    Returns exponent, coefficient, and R^2. Needs >= 2 positive points.
    """
    pairs = [(xi, yi) for xi, yi in zip(x, y)
             if xi is not None and yi is not None and xi > 0 and yi > 0]
    if len(pairs) < 2:
        return {"exponent": None, "coefficient": None, "r_squared": None, "n": len(pairs)}

    lx = np.log(np.array([p[0] for p in pairs], dtype=float))
    ly = np.log(np.array([p[1] for p in pairs], dtype=float))
    b, log_a = np.polyfit(lx, ly, 1)
    pred = b * lx + log_a
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "exponent": round(float(b), 4),
        "coefficient": round(float(math.exp(log_a)), 6),
        "r_squared": round(r2, 4) if r2 is not None else None,
        "n": len(pairs),
    }


def summarize_traces(traces: List[LatencyTrace]) -> Dict:
    """Aggregate a sweep of traces into scaling fits keyed on input length."""
    by_len = sorted(traces, key=lambda t: t.input_tokens)
    lengths = [t.input_tokens for t in by_len]
    ttfts = [t.ttft_s for t in by_len]
    itls = [t.mean_itl_s for t in by_len]
    return {
        "per_length": [t.to_dict() for t in by_len],
        "prefill_scaling": fit_power_law(lengths, ttfts),     # TTFT vs input length
        "decode_scaling": fit_power_law(lengths, itls),       # ITL vs context length
    }
