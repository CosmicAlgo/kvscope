"""
src/api/profiler.py: orchestration for the black-box API profiler.

Two experiments:

  * :class:`LatencyProfiler` — sweeps input length and records TTFT / ITL /
    throughput, then fits the prefill and decode scaling exponents.
  * :class:`NeedleProfiler` — needle-in-a-haystack recall over a
    (context length x depth) grid, i.e. effective context.

Both take a :class:`~src.api.client.ChatClient` and an optional tokenizer (for
exact token counting — e.g. the MiniMax-M2 tokenizer as an M3 stand-in).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.api.client import ChatClient
from src.api.metrics import LatencyTrace, summarize_traces
from src.api.prompts import build_length_prompt, build_needle_prompt, count_tokens


def load_standin_tokenizer(model: Optional[str]):
    """Best-effort tokenizer load (returns None if transformers/weights absent)."""
    if not model:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return None


class LatencyProfiler:
    def __init__(self, client: ChatClient, tokenizer=None):
        self.client = client
        self.tokenizer = tokenizer

    def run_one(self, input_tokens: int, max_tokens: int = 128) -> LatencyTrace:
        messages = build_length_prompt(input_tokens, tokenizer=self.tokenizer)
        actual_in = count_tokens(messages[0]["content"], self.tokenizer)
        trace = LatencyTrace(input_tokens=actual_in, t_send=time.perf_counter(),
                             label=f"in={input_tokens}")
        for event in self.client.stream_chat(messages, max_tokens=max_tokens):
            trace.token_timestamps.append(event.ts)
        trace.t_end = time.perf_counter()
        trace.output_tokens = len(trace.token_timestamps)
        return trace

    def sweep(self, lengths: List[int], max_tokens: int = 128,
              repeats: int = 1, verbose: bool = True) -> Dict:
        traces: List[LatencyTrace] = []
        for n in lengths:
            for r in range(repeats):
                t = self.run_one(n, max_tokens=max_tokens)
                traces.append(t)
                if verbose:
                    print(f"[latency] in~{n:>8} tok  TTFT={_fmt(t.ttft_s)}s  "
                          f"ITL={_fmt(t.mean_itl_s)}s  out={t.output_tokens}")
        summary = summarize_traces(traces)
        summary["config"] = {
            "model": self.client.model,
            "provider": self.client.provider,
            "lengths": lengths,
            "max_tokens": max_tokens,
            "repeats": repeats,
        }
        return summary


@dataclass
class NeedleResult:
    context_tokens: int
    depth_fraction: float
    found: bool
    expected: str
    response_excerpt: str


class NeedleProfiler:
    def __init__(self, client: ChatClient, tokenizer=None):
        self.client = client
        self.tokenizer = tokenizer

    def run_one(self, context_tokens: int, depth_fraction: float,
                max_tokens: int = 32) -> NeedleResult:
        messages, expected = build_needle_prompt(
            context_tokens, depth_fraction, tokenizer=self.tokenizer)
        text, _ = self.client.complete(messages, max_tokens=max_tokens)
        return NeedleResult(
            context_tokens=context_tokens,
            depth_fraction=round(depth_fraction, 3),
            found=expected in text,
            expected=expected,
            response_excerpt=text.strip()[:80],
        )

    def grid(self, lengths: List[int], depths: List[float],
             max_tokens: int = 32, verbose: bool = True) -> Dict:
        results: List[NeedleResult] = []
        for n in lengths:
            for d in depths:
                r = self.run_one(n, d, max_tokens=max_tokens)
                results.append(r)
                if verbose:
                    mark = "OK " if r.found else "MISS"
                    print(f"[needle] ctx~{n:>8} depth={d:<4} {mark} "
                          f"(want {r.expected}) :: {r.response_excerpt}")
        total = len(results)
        found = sum(1 for r in results if r.found)
        return {
            "config": {
                "model": self.client.model,
                "provider": self.client.provider,
                "lengths": lengths,
                "depths": depths,
            },
            "overall_recall": round(found / total, 4) if total else None,
            "grid": [vars(r) for r in results],
        }


def _fmt(x: Optional[float]) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
