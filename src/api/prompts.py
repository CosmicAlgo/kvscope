"""
src/api/prompts.py: synthetic prompt construction for the API profiler.

Builds prompts of a target token length and needle-in-a-haystack probes. Token
counting uses a real tokenizer when one is supplied (e.g. the MiniMax-M2
tokenizer as an M3 stand-in); otherwise it falls back to a ~4-chars/token
heuristic, which is good enough for sweep *shape* even if absolute counts drift.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

_CHARS_PER_TOKEN = 4  # rough heuristic when no tokenizer is available

# Varied filler so the context does not trivially compress.
_FILLER_SENTENCES = [
    "The quarterly logistics report noted a steady increase in regional throughput.",
    "Researchers observed that the catalyst improved yield under moderate pressure.",
    "In the third chapter, the protagonist reconsiders the meaning of the journey.",
    "Network latency remained within tolerance despite the elevated request volume.",
    "The committee deferred the budget vote pending a revised cost projection.",
    "Sedimentary layers in the canyon record millions of years of climate change.",
    "A balanced portfolio diversifies across sectors to dampen idiosyncratic risk.",
    "The compiler emitted a warning about an unused variable in the parser module.",
    "Migratory patterns shifted earlier in the season than historical averages.",
    "The orchestra rehearsed the second movement until the phrasing felt natural.",
]


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)


def make_filler(n_tokens: int, tokenizer=None, seed: int = 0) -> str:
    """Produce filler text of approximately ``n_tokens`` tokens."""
    rng = random.Random(seed)
    if tokenizer is not None:
        # Grow until the tokenizer agrees we have enough, then trim.
        parts: List[str] = []
        while count_tokens(" ".join(parts), tokenizer) < n_tokens:
            parts.append(rng.choice(_FILLER_SENTENCES))
        text = " ".join(parts)
        ids = tokenizer.encode(text)
        if len(ids) > n_tokens:
            text = tokenizer.decode(ids[:n_tokens])
        return text
    # Heuristic path: target a character budget.
    target_chars = n_tokens * _CHARS_PER_TOKEN
    parts = []
    total = 0
    while total < target_chars:
        s = rng.choice(_FILLER_SENTENCES)
        parts.append(s)
        total += len(s) + 1
    return " ".join(parts)[:target_chars]


def build_length_prompt(
    n_input_tokens: int,
    question: str = "Summarize the document above in one sentence.",
    tokenizer=None,
) -> List[Dict[str, str]]:
    """Chat messages whose user content is ~``n_input_tokens`` tokens long."""
    filler = make_filler(max(16, n_input_tokens - count_tokens(question, tokenizer) - 8), tokenizer)
    content = f"{filler}\n\n{question}"
    return [{"role": "user", "content": content}]


def build_needle_prompt(
    n_context_tokens: int,
    depth_fraction: float,
    tokenizer=None,
    seed: int = 0,
) -> Tuple[List[Dict[str, str]], str]:
    """Construct a needle-in-a-haystack probe.

    A unique fact is embedded at ``depth_fraction`` (0.0 = start, 1.0 = end) of a
    long filler context, then the model is asked to retrieve it. Returns the chat
    messages and the expected answer string.

    This measures *effective* context: MSA can run fast at 1M tokens, but does it
    still recall a fact buried at 80% depth?
    """
    rng = random.Random(seed)
    magic = rng.randint(10_000, 99_999)
    topic = rng.choice(["the Aurora project", "the Helix dataset", "the Onyx release",
                         "the Vega migration", "the Cobalt experiment"])
    needle = f"Important: the secret access code for {topic} is {magic}."

    depth_fraction = min(max(depth_fraction, 0.0), 1.0)
    before_tokens = max(8, int(n_context_tokens * depth_fraction))
    after_tokens = max(8, n_context_tokens - before_tokens)
    before = make_filler(before_tokens, tokenizer, seed=seed + 1)
    after = make_filler(after_tokens, tokenizer, seed=seed + 2)

    question = f"What is the secret access code for {topic}? Answer with the number only."
    content = f"{before}\n\n{needle}\n\n{after}\n\n{question}"
    return [{"role": "user", "content": content}], str(magic)
