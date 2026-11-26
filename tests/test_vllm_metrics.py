"""Offline tests for the vLLM Prometheus metrics parser (no server needed)."""

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRAPER = os.path.join(_HERE, "..", "experiments", "vllm_comparison", "scrape_vllm_metrics.py")


def _load_scraper():
    spec = importlib.util.spec_from_file_location("scrape_vllm_metrics", _SCRAPER)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve cls.__module__ under
    # `from __future__ import annotations`.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SAMPLE = """\
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="MiniMax-M3"} 0.42
vllm:num_requests_running{model_name="MiniMax-M3"} 3
vllm:num_requests_waiting{model_name="MiniMax-M3"} 1
vllm:generation_tokens_total{model_name="MiniMax-M3"} 12345
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model_name="MiniMax-M3"} 4.0
vllm:time_to_first_token_seconds_count{model_name="MiniMax-M3"} 8
"""


def test_parse_prometheus_basic():
    scraper = _load_scraper()
    samples = scraper.parse_prometheus(SAMPLE)
    names = {s.name for s in samples}
    assert "vllm:gpu_cache_usage_perc" in names
    usage = next(s for s in samples if s.name == "vllm:gpu_cache_usage_perc")
    assert usage.value == pytest.approx(0.42)
    assert usage.labels["model_name"] == "MiniMax-M3"


def test_summarize_kv_metrics():
    scraper = _load_scraper()
    summary = scraper.summarize_kv_metrics(scraper.parse_prometheus(SAMPLE))
    assert summary["kv_cache_usage_frac"] == pytest.approx(0.42)
    assert summary["requests_running"] == pytest.approx(3)
    assert summary["requests_waiting"] == pytest.approx(1)
    assert summary["generation_tokens_total"] == pytest.approx(12345)
    # histogram average = sum / count = 4.0 / 8 = 0.5
    assert summary["ttft_seconds_avg"] == pytest.approx(0.5)


def test_summarize_handles_missing_metrics():
    scraper = _load_scraper()
    summary = scraper.summarize_kv_metrics(scraper.parse_prometheus("# nothing\n"))
    assert summary["kv_cache_usage_frac"] is None
    assert summary["ttft_seconds_avg"] is None
