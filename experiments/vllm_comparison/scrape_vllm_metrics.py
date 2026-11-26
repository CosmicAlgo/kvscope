#!/usr/bin/env python3
"""
scrape_vllm_metrics.py: read KV-cache + serving metrics from a vLLM server.

vLLM serves models with PagedAttention and exposes a Prometheus ``/metrics``
endpoint. That endpoint is the closest thing to *KV-cache profiling through a
server*: ``vllm:gpu_cache_usage_perc`` reports the fraction of KV blocks in use,
which — paired with the API behavioural profiler's latency curves — lets us
characterise MSA at long context without touching the model's internals.

This module:
  * parses Prometheus text exposition format (no external deps),
  * summarises the KV-cache and throughput gauges/counters/histograms,
  * can poll the endpoint over time and dump a JSON time series.

Usage:
    python experiments/vllm_comparison/scrape_vllm_metrics.py \
        --url http://localhost:8000/metrics --interval 1 --duration 60 \
        --output results/api/vllm_m3_kv_timeseries.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Make ``import src.*`` work no matter where this script is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Metric name → human label for the KV/serving subset we care about.
KV_GAUGES = {
    "vllm:gpu_cache_usage_perc": "kv_cache_usage_frac",
    "vllm:cpu_cache_usage_perc": "cpu_cache_usage_frac",
    "vllm:gpu_prefix_cache_hit_rate": "prefix_cache_hit_rate",
    "vllm:num_requests_running": "requests_running",
    "vllm:num_requests_waiting": "requests_waiting",
}
TOKEN_COUNTERS = {
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
}
LATENCY_HISTOGRAMS = {
    "vllm:time_to_first_token_seconds": "ttft_seconds",
    "vllm:time_per_output_token_seconds": "tpot_seconds",
}


@dataclass
class Sample:
    name: str
    labels: Dict[str, str]
    value: float


def parse_prometheus(text: str) -> List[Sample]:
    """Parse Prometheus text exposition format into a flat list of samples."""
    samples: List[Sample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, labels, value = _parse_line(line)
        if name is None:
            continue
        try:
            val = float(value)
        except ValueError:
            continue
        samples.append(Sample(name=name, labels=labels, value=val))
    return samples


def _parse_line(line: str) -> Tuple[Optional[str], Dict[str, str], str]:
    # Forms: name value   |   name{l1="a",l2="b"} value
    if "{" in line:
        name, rest = line.split("{", 1)
        label_str, _, value = rest.partition("}")
        value = value.strip()
        return name.strip(), _parse_labels(label_str), value
    parts = line.split()
    if len(parts) < 2:
        return None, {}, ""
    return parts[0], {}, parts[1]


def _parse_labels(s: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for pair in s.split(","):
        if "=" not in pair:
            continue
        k, _, v = pair.partition("=")
        labels[k.strip()] = v.strip().strip('"')
    return labels


def _first(samples: List[Sample], name: str) -> Optional[float]:
    for s in samples:
        if s.name == name:
            return s.value
    return None


def _hist_avg(samples: List[Sample], base: str) -> Optional[float]:
    """Average = _sum / _count for a Prometheus histogram (summed over labels)."""
    total = sum(s.value for s in samples if s.name == base + "_sum")
    count = sum(s.value for s in samples if s.name == base + "_count")
    return (total / count) if count > 0 else None


def summarize_kv_metrics(samples: List[Sample]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for metric, label in KV_GAUGES.items():
        out[label] = _first(samples, metric)
    for metric, label in TOKEN_COUNTERS.items():
        out[label] = _first(samples, metric)
    for metric, label in LATENCY_HISTOGRAMS.items():
        out[f"{label}_avg"] = _hist_avg(samples, metric)
    return out


def fetch(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def poll(url: str, interval: float, duration: float) -> List[Dict]:
    """Sample the endpoint every ``interval`` s for ``duration`` s."""
    series: List[Dict] = []
    t_end = time.time() + duration
    while time.time() < t_end:
        t = time.time()
        try:
            samples = parse_prometheus(fetch(url))
            row = summarize_kv_metrics(samples)
            row["t"] = round(t, 3)
            series.append(row)
            usage = row.get("kv_cache_usage_frac")
            print(f"[vllm-metrics] kv_usage={usage}  running={row.get('requests_running')}")
        except Exception as e:  # pragma: no cover - network dependent
            print(f"[vllm-metrics] fetch error: {e}")
        time.sleep(max(0.0, interval - (time.time() - t)))
    return series


def main() -> int:
    p = argparse.ArgumentParser(description="Scrape vLLM KV-cache / serving metrics")
    p.add_argument("--url", default="http://localhost:8000/metrics")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    series = poll(args.url, args.interval, args.duration)
    peak = max((r.get("kv_cache_usage_frac") or 0.0 for r in series), default=0.0)
    print(f"\n[vllm-metrics] peak KV cache usage: {peak:.3f} over {len(series)} samples")
    if args.output:
        from src.profiler.json_utils import dump_json
        dump_json({"peak_kv_cache_usage_frac": peak, "series": series}, args.output)
        print(f"[vllm-metrics] saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
