"""Offline tests for synthetic prompt construction + client config (no network)."""

import pytest

from src.api.client import PROVIDERS, ChatClient
from src.api.prompts import (
    build_length_prompt,
    build_needle_prompt,
    count_tokens,
    make_filler,
)


def test_count_tokens_heuristic():
    assert count_tokens("") == 1  # clamped to >= 1
    assert count_tokens("a" * 40) == 10  # 40 chars / 4


def test_make_filler_targets_length_heuristic():
    text = make_filler(500)
    # Heuristic targets ~4 chars/token; allow generous tolerance on the estimate.
    est = count_tokens(text)
    assert 400 <= est <= 600


def test_build_length_prompt_returns_user_message():
    msgs = build_length_prompt(2000)
    assert isinstance(msgs, list) and msgs[0]["role"] == "user"
    assert len(msgs[0]["content"]) > 0


def test_build_needle_prompt_embeds_answer():
    msgs, expected = build_needle_prompt(4000, depth_fraction=0.5, seed=7)
    content = msgs[0]["content"]
    assert expected in content          # the needle is actually present
    assert expected.isdigit()
    assert "secret access code" in content


def test_needle_depth_changes_position():
    early, _ = build_needle_prompt(4000, 0.1, seed=1)
    late, _ = build_needle_prompt(4000, 0.9, seed=1)
    # Needle sits later in the deeper-depth prompt.
    idx_early = early[0]["content"].index("secret access code")
    idx_late = late[0]["content"].index("secret access code")
    assert idx_late > idx_early


def test_client_provider_presets_and_key_resolution(monkeypatch):
    monkeypatch.delenv("KVSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("KVSCOPE_API_BASE_URL", raising=False)
    monkeypatch.delenv("KVSCOPE_API_MODEL", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-123")
    c = ChatClient(provider="minimax")
    assert c.base_url == PROVIDERS["minimax"]["base_url"]
    assert c.model == "MiniMax-M3"
    assert c.api_key == "test-key-123"


def test_client_unknown_provider_raises():
    with pytest.raises(ValueError):
        ChatClient(provider="nope")
