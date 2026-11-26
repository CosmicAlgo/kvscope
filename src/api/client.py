"""
src/api/client.py: minimal, dependency-free streaming chat client.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint (MiniMax platform,
OpenRouter, vLLM/SGLang servers, ...). Uses only the standard library so it runs
anywhere — streaming Server-Sent Events are parsed by hand.

Credentials and endpoint come from a provider preset plus environment variables;
nothing is ever hard-coded. Never commit API keys.

    export KVSCOPE_API_KEY=sk-...           # or MINIMAX_API_KEY / OPENROUTER_API_KEY
    kvscope api smoke --provider minimax --model MiniMax-M3
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

# Provider presets: base_url, default model, and the env var checked first for
# the key. KVSCOPE_API_KEY is always honoured as a fallback.
PROVIDERS: Dict[str, Dict[str, str]] = {
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-M3",
        "key_env": "MINIMAX_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "minimax/minimax-m3",
        "key_env": "OPENROUTER_API_KEY",
    },
    # Generic OpenAI-compatible server (e.g. local vLLM/SGLang). Override base_url.
    "openai-compatible": {
        "base_url": "http://localhost:8000/v1",
        "model": "minimax-m3",
        "key_env": "KVSCOPE_API_KEY",
    },
}


class APIError(RuntimeError):
    pass


@dataclass
class StreamEvent:
    text: str          # incremental content delta
    ts: float          # perf_counter() timestamp when the delta arrived


class ChatClient:
    """Thin OpenAI-compatible chat client with streaming token timing."""

    def __init__(
        self,
        provider: str = "minimax",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 600.0,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Known: {sorted(PROVIDERS)}")
        preset = PROVIDERS[provider]
        self.provider = provider
        self.base_url = (base_url or os.environ.get("KVSCOPE_API_BASE_URL")
                         or preset["base_url"]).rstrip("/")
        self.model = model or os.environ.get("KVSCOPE_API_MODEL") or preset["model"]
        self.api_key = (api_key
                        or os.environ.get(preset["key_env"])
                        or os.environ.get("KVSCOPE_API_KEY"))
        self.timeout = timeout

    # ── internals ─────────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise APIError(
                "No API key found. Set one of KVSCOPE_API_KEY / "
                f"{PROVIDERS[self.provider]['key_env']} in the environment."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages, max_tokens, temperature, stream, extra) -> bytes:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if extra:
            body.update(extra)
        return json.dumps(body).encode("utf-8")

    def _request(self, payload: bytes):
        url = f"{self.base_url}/chat/completions"
        return urllib.request.Request(url, data=payload, headers=self._headers(), method="POST")

    # ── streaming ───────────────────────────────────────────────────────────────

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> Iterator[StreamEvent]:
        """Yield :class:`StreamEvent` per content delta, with arrival timestamps.

        The HTTP request is issued lazily on the first iteration, so the caller
        should record its send time immediately before consuming the generator
        to measure time-to-first-token accurately.
        """
        req = self._request(self._payload(messages, max_tokens, temperature, True, extra))
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:  # pragma: no cover - network dependent
            detail = e.read().decode("utf-8", "replace")[:500]
            raise APIError(f"HTTP {e.code} from {self.base_url}: {detail}") from e
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield StreamEvent(text=delta, ts=time.perf_counter())

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> Tuple[str, Dict]:
        """Non-streaming completion. Returns (text, usage_dict)."""
        req = self._request(self._payload(messages, max_tokens, temperature, False, extra))
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:  # pragma: no cover - network dependent
            detail = e.read().decode("utf-8", "replace")[:500]
            raise APIError(f"HTTP {e.code} from {self.base_url}: {detail}") from e
        with resp:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
        text = obj["choices"][0]["message"]["content"]
        usage = obj.get("usage", {})
        return text, usage
