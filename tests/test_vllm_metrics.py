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


CURRENT = """\
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:prompt_tokens_total{engine="0",model_name="m"} 1000
vllm:generation_tokens_total{engine="0",model_name="m"} 500
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 800
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 200
vllm:request_time_per_output_token_seconds_sum{engine="0",model_name="m"} 2.0
vllm:request_time_per_output_token_seconds_count{engine="0",model_name="m"} 40
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m"} 100.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m"} 100
"""

# Ten seconds later: 10 more requests whose TTFTs sum to 50 s (5 s each).
LATER = CURRENT.replace(" 100.0\n", " 150.0\n").replace("_count{engine=\"0\",model_name=\"m\"} 100\n",
                                                        "_count{engine=\"0\",model_name=\"m\"} 110\n") \
               .replace("generation_tokens_total{engine=\"0\",model_name=\"m\"} 500",
                        "generation_tokens_total{engine=\"0\",model_name=\"m\"} 700") \
               .replace("prefix_cache_queries_total{engine=\"0\",model_name=\"m\"} 800",
                        "prefix_cache_queries_total{engine=\"0\",model_name=\"m\"} 900") \
               .replace("prefix_cache_hits_total{engine=\"0\",model_name=\"m\"} 200",
                        "prefix_cache_hits_total{engine=\"0\",model_name=\"m\"} 290")


def test_current_metric_names_and_provenance():
    scraper = _load_scraper()
    s = scraper.summarize_kv_metrics(scraper.parse_prometheus(CURRENT))
    assert s["kv_cache_usage_frac"] == pytest.approx(0.25)
    assert s["metric_names"]["kv_cache_usage_frac"] == "vllm:kv_cache_usage_perc"
    assert s["tpot_seconds_avg"] == pytest.approx(0.05)
    assert s["metric_names"]["tpot_seconds"] == "vllm:request_time_per_output_token_seconds"
    assert s["prefix_cache_hit_rate"] == pytest.approx(0.25)


def test_old_metric_name_still_read():
    scraper = _load_scraper()
    s = scraper.summarize_kv_metrics(scraper.parse_prometheus(SAMPLE))
    assert s["metric_names"]["kv_cache_usage_frac"] == "vllm:gpu_cache_usage_perc"


def test_window_average_uses_deltas_not_lifetime_totals():
    scraper = _load_scraper()
    a = scraper.summarize_kv_metrics(scraper.parse_prometheus(CURRENT))
    b = scraper.summarize_kv_metrics(scraper.parse_prometheus(LATER))
    w = scraper.summarize_window(a, b, dt_s=10.0)
    # Lifetime average is 150/110 = 1.36 s; the interval's requests took 5 s each.
    assert b["ttft_seconds_avg"] == pytest.approx(150 / 110)
    assert w["ttft_seconds_avg_window"] == pytest.approx(5.0)
    assert w["generation_tokens_per_s_window"] == pytest.approx(20.0)   # 200 tokens / 10 s
    assert w["prefix_cache_hit_rate_window"] == pytest.approx(0.9)      # 90 hits / 100 queries


def test_window_after_counter_reset_is_none():
    scraper = _load_scraper()
    a = scraper.summarize_kv_metrics(scraper.parse_prometheus(LATER))
    b = scraper.summarize_kv_metrics(scraper.parse_prometheus(CURRENT))  # counters went down
    w = scraper.summarize_window(a, b, dt_s=10.0)
    assert w["ttft_seconds_avg_window"] is None
    assert w["generation_tokens_per_s_window"] is None
