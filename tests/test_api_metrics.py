"""Offline tests for the API profiler timing math (no network)."""

import pytest

from src.api.metrics import LatencyTrace, fit_power_law, summarize_traces


def test_latency_trace_derived_metrics():
    t = LatencyTrace(input_tokens=100, t_send=0.0,
                     token_timestamps=[1.0, 1.5, 2.0, 2.7], t_end=3.0)
    assert t.ttft_s == pytest.approx(1.0)
    assert t.inter_token_latencies_s == pytest.approx([0.5, 0.5, 0.7])
    assert t.mean_itl_s == pytest.approx(1.7 / 3)
    assert t.decode_tokens_per_s == pytest.approx(3 / 1.7)
    assert t.total_latency_s == pytest.approx(3.0)


def test_latency_trace_empty_is_safe():
    t = LatencyTrace(input_tokens=10, t_send=0.0)
    assert t.ttft_s is None
    assert t.mean_itl_s is None
    assert t.decode_tokens_per_s is None
    d = t.to_dict()
    assert d["ttft_s"] is None


def test_fit_power_law_recovers_quadratic():
    x = [1, 2, 4, 8, 16]
    y = [3 * xi ** 2 for xi in x]
    fit = fit_power_law(x, y)
    assert fit["exponent"] == pytest.approx(2.0, abs=1e-6)
    assert fit["coefficient"] == pytest.approx(3.0, rel=1e-4)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_fit_power_law_flat_decode():
    # Constant per-token latency -> exponent ~ 0 (ideal sparse decode signature).
    x = [1000, 4000, 16000, 64000]
    y = [0.02, 0.02, 0.02, 0.02]
    fit = fit_power_law(x, y)
    assert fit["exponent"] == pytest.approx(0.0, abs=1e-6)


def test_fit_power_law_insufficient_points():
    assert fit_power_law([1], [2])["exponent"] is None


def test_summarize_traces_shape():
    traces = [
        LatencyTrace(input_tokens=1000, t_send=0.0,
                     token_timestamps=[0.5, 0.7, 0.9], t_end=1.0),
        LatencyTrace(input_tokens=4000, t_send=0.0,
                     token_timestamps=[1.0, 1.2, 1.4], t_end=1.5),
    ]
    summary = summarize_traces(traces)
    assert "prefill_scaling" in summary
    assert "decode_scaling" in summary
    assert len(summary["per_length"]) == 2
