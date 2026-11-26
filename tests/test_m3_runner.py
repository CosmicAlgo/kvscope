"""Weight-free tests for the MiniMax M3 runner wiring.

Skipped entirely when ``transformers`` is unavailable (e.g. lean CI), since the
runner imports it at module load. No model weights or GPU are required.
"""

import pytest

pytest.importorskip("transformers")

import torch  # noqa: E402

from src.models.minimax_m3_runner import MODEL_TYPE, MiniMaxM3Runner, _avg  # noqa: E402
from src.profiler.kv_tracer import KVCacheTracer  # noqa: E402


def test_model_type_registered():
    assert MODEL_TYPE == "minimax_m3"
    assert MODEL_TYPE in KVCacheTracer.MODEL_TYPES


def test_avg_helper():
    assert _avg([1, 2, 3]) == 2.0
    assert _avg([]) == 0.0


def test_multi_gpu_autodetect(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    r = MiniMaxM3Runner("dummy/path")
    assert r.multi_gpu is True

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    r2 = MiniMaxM3Runner("dummy/path")
    assert r2.multi_gpu is False

    # Explicit override wins over autodetect.
    r3 = MiniMaxM3Runner("dummy/path", multi_gpu=True)
    assert r3.multi_gpu is True


def test_quant_config_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert MiniMaxM3Runner("p").quant_config is None
    assert MiniMaxM3Runner("p", load_in_8bit=True).quant_config is not None
    assert MiniMaxM3Runner("p", load_in_4bit=True).quant_config is not None


def test_standin_flag_recorded(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    r = MiniMaxM3Runner("/models/minimax_m2", is_standin=True)
    assert r.is_standin is True
    assert r.adapter.model_type == "minimax_m3"
