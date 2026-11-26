#!/usr/bin/env python3
"""
scrape_vllm_metrics.py: read KV-cache + serving metrics from a vLLM server.

vLLM serves models with PagedAttention and exposes a Prometheus ``/metrics``
endpoint. That endpoint is the closest thing to *KV-cache profiling through a
server*: ``vllm:kv_cache_usage_perc`` (``vllm:gpu_cache_usage_perc`` in older
vLLM) reports the fraction of KV blocks in use. Paired with the API profiler's
latency curves, it characterises serving behaviour without touching the
model's internals.

This module:
  * parses Prometheus text exposition format (no external deps),
  * summarises the KV-cache and throughput gauges/counters/histograms,
  * accepts current and older vLLM metric names and records which one it found,
  * when polling, reports latency averages and rates for each interval
    (Prometheus counters and histogram sums are cumulative since server start).

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

# Output label -> candidate metric names, newest first. Names checked against
# https://docs.vllm.ai/en/stable/usage/metrics.html (November 2026); older
# names are kept so the scraper also works against older vLLM versions.
# Counters are exposed with a ``_total`` suffix in the text format.
GAUGES = {
    "kv_cache_usage_frac": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "cpu_cache_usage_frac": ["vllm:cpu_cache_usage_perc"],
    "requests_running": ["vllm:num_requests_running"],
    "requests_waiting": ["vllm:num_requests_waiting"],
}
COUNTERS = {
    "prompt_tokens_total": ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
    "generation_tokens_total": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
    "prefix_cache_queries_total": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"],
    "prefix_cache_hits_total": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
}
HISTOGRAMS = {
    "ttft_seconds": ["vllm:time_to_first_token_seconds"],
    "tpot_seconds": ["vllm:request_time_per_output_token_seconds",
                     "vllm:time_per_output_token_seconds"],
    "itl_seconds": ["vllm:inter_token_latency_seconds"],
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


def _first(samples: List[Sample], names: List[str]) -> Tuple[Optional[float], Optional[str]]:
    """Value of the first candidate name present (first label set), and that name."""
    for name in names:
        for s in samples:
            if s.name == name:
                return s.value, name
    return None, None


def _hist_totals(samples: List[Sample], names: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Cumulative (_sum, _count) for the first candidate histogram present,
    summed over label sets."""
    for base in names:
        sums = [s.value for s in samples if s.name == base + "_sum"]
        counts = [s.value for s in samples if s.name == base + "_count"]
        if counts:
            return sum(sums), sum(counts), base
    return None, None, None


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    return (num / den) if (num is not None and den) else None


def summarize_kv_metrics(samples: List[Sample]) -> Dict[str, Optional[float]]:
    """One scrape on its own. Latency ``*_avg`` values here are averages over the
    server's whole lifetime (cumulative _sum / _count)."""
    out: Dict = {}
    used: Dict[str, str] = {}
    for label, names in {**GAUGES, **COUNTERS}.items():
        out[label], name = _first(samples, names)
        if name:
            used[label] = name
    for label, names in HISTOGRAMS.items():
        total, count, name = _hist_totals(samples, names)
        out[f"{label}_sum"], out[f"{label}_count"] = total, count
        out[f"{label}_avg"] = _ratio(total, count)
        if name:
            used[label] = name
    out["prefix_cache_hit_rate"] = _ratio(out["prefix_cache_hits_total"],
                                          out["prefix_cache_queries_total"])
    out["metric_names"] = used
    return out


def _delta(cur: Dict, prev: Dict, key: str) -> Optional[float]:
    a, b = cur.get(key), prev.get(key)
    if a is None or b is None or a < b:   # a < b: server restarted, counters reset
        return None
    return a - b


def summarize_window(prev: Dict, cur: Dict, dt_s: float) -> Dict[str, Optional[float]]:
    """Averages and rates for the interval between two scrapes.

    Histogram averages use (delta _sum) / (delta _count), so a value reflects
    only requests that completed in this interval.
    """
    out: Dict[str, Optional[float]] = {}
    for label in HISTOGRAMS:
        out[f"{label}_avg_window"] = _ratio(_delta(cur, prev, f"{label}_sum"),
                                            _delta(cur, prev, f"{label}_count"))
    for label in ("prompt_tokens_total", "generation_tokens_total"):
        d = _delta(cur, prev, label)
        out[label.replace("_total", "_per_s_window")] = (d / dt_s) if (d is not None and dt_s > 0) else None
    out["prefix_cache_hit_rate_window"] = _ratio(_delta(cur, prev, "prefix_cache_hits_total"),
                                                 _delta(cur, prev, "prefix_cache_queries_total"))
    return out


def fetch(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def poll(url: str, interval: float, duration: float) -> List[Dict]:
    """Sample the endpoint every ``interval`` s for ``duration`` s."""
    series: List[Dict] = []
    prev: Optional[Dict] = None
    t_end = time.time() + duration
    while time.time() < t_end:
        t = time.time()
        try:
            samples = parse_prometheus(fetch(url))
            row = summarize_kv_metrics(samples)
            row["t"] = round(t, 3)
            if prev is not None:
                row.update(summarize_window(prev, row, t - prev["t"]))
            prev = row
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
