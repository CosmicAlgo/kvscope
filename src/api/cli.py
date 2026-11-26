"""
src/api/cli.py: subcommands for ``kvscope api ...``.

    kvscope api smoke    --provider minimax --model MiniMax-M3
    kvscope api latency  --lengths 1000,4000,16000,64000 --max-tokens 128 -o results/api/m3_latency.json
    kvscope api needle   --lengths 16000,64000,256000 --depths 0,0.25,0.5,0.75,1.0 -o results/api/m3_needle.json

Credentials come from the environment (KVSCOPE_API_KEY / MINIMAX_API_KEY /
OPENROUTER_API_KEY). Never pass keys on the command line.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from src.api.client import ChatClient
from src.api.profiler import LatencyProfiler, NeedleProfiler, load_standin_tokenizer
from src.profiler.json_utils import dump_json


def _int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _make_client(args) -> ChatClient:
    return ChatClient(provider=args.provider, base_url=args.base_url, model=args.model)


def _cmd_smoke(args) -> int:
    client = _make_client(args)
    print(f"[smoke] provider={client.provider} base_url={client.base_url} model={client.model}")
    text, usage = client.complete(
        [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=8)
    print(f"[smoke] response: {text!r}")
    if usage:
        print(f"[smoke] usage: {usage}")
    return 0


def _cmd_latency(args) -> int:
    client = _make_client(args)
    tok = load_standin_tokenizer(args.tokenizer)
    if args.tokenizer and tok is None:
        print(f"[latency] tokenizer '{args.tokenizer}' unavailable; using char heuristic")
    prof = LatencyProfiler(client, tokenizer=tok)
    summary = prof.sweep(args.lengths, max_tokens=args.max_tokens, repeats=args.repeats)
    pf = summary["prefill_scaling"]
    df = summary["decode_scaling"]
    print(f"\n[latency] prefill TTFT scaling exponent b={pf['exponent']} (R^2={pf['r_squared']})")
    print(f"[latency] decode  ITL  scaling exponent b={df['exponent']} (R^2={df['r_squared']})")
    print("          b~0 flat (ideal sparse) | b~1 linear | b~2 quadratic")
    if args.output:
        dump_json(summary, args.output)
        print(f"[latency] saved -> {args.output}")
    return 0


def _cmd_needle(args) -> int:
    client = _make_client(args)
    tok = load_standin_tokenizer(args.tokenizer)
    prof = NeedleProfiler(client, tokenizer=tok)
    result = prof.grid(args.lengths, args.depths, max_tokens=args.max_tokens)
    print(f"\n[needle] overall recall: {result['overall_recall']}")
    if args.output:
        dump_json(result, args.output)
        print(f"[needle] saved -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kvscope api",
                                description="Black-box API behavioural profiler")
    p.add_argument("--provider", default="minimax",
                   help="minimax | openrouter | openai-compatible")
    p.add_argument("--base-url", default=None, help="override endpoint base URL")
    p.add_argument("--model", default=None, help="model id (defaults to provider preset)")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer id for exact token counts (e.g. MiniMaxAI/MiniMax-M2)")
    sub = p.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("smoke", help="single round-trip sanity check")

    lat = sub.add_parser("latency", help="prefill/decode latency sweep vs input length")
    lat.add_argument("--lengths", type=_int_list, default=[1000, 4000, 16000, 64000],
                     help="comma-separated input token lengths")
    lat.add_argument("--max-tokens", type=int, default=128)
    lat.add_argument("--repeats", type=int, default=1)
    lat.add_argument("-o", "--output", default=None)

    ndl = sub.add_parser("needle", help="needle-in-a-haystack recall grid")
    ndl.add_argument("--lengths", type=_int_list, default=[16000, 64000, 256000])
    ndl.add_argument("--depths", type=_float_list, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ndl.add_argument("--max-tokens", type=int, default=32)
    ndl.add_argument("-o", "--output", default=None)

    return p


def run(argv: Optional[List[str]] = None) -> int:
    # Strip a leading "--" that argparse.REMAINDER may forward from the parent.
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = build_parser().parse_args(argv)
    if args.subcommand == "smoke":
        return _cmd_smoke(args)
    if args.subcommand == "latency":
        return _cmd_latency(args)
    if args.subcommand == "needle":
        return _cmd_needle(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
