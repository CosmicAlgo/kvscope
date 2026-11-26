"""
src/cli.py: KVScope command-line entry point.

Wired to the ``kvscope`` console script declared in ``pyproject.toml``
(``kvscope = "src.cli:main"``). Subcommands:

    kvscope list                 # list every known architecture + metadata
    kvscope info <model_type>    # detailed adapter info + analytical density
    kvscope api ...              # black-box API behavioural profiler (see src/api)

The ``list`` / ``info`` commands are import-light (no torch / CUDA needed) so they
work anywhere, including CI.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _cmd_list(args: argparse.Namespace) -> int:
    from src.profiler.adapters import ADAPTERS

    rows = []
    for mt, a in sorted(ADAPTERS.items()):
        weights = "public" if a.weights_public else "PENDING"
        ctx = f"{a.max_context:,}" if a.max_context else "-"
        rows.append((mt, a.kv_strategy, weights, ctx, a.testable_model or "-"))

    widths = [max(len(str(r[i])) for r in rows + [("model_type", "kv_strategy", "weights", "max_ctx", "test_model")]) for i in range(5)]
    header = ("model_type", "kv_strategy", "weights", "max_ctx", "test_model")
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(line)
    print("  ".join("-" * widths[i] for i in range(5)))
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(5)))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from src.profiler.adapters import (
        ADAPTERS,
        analytical_density_kb_per_token_per_layer,
        get_adapter,
    )

    if args.model_type not in ADAPTERS:
        print(f"Unknown model_type '{args.model_type}'. Known: {sorted(ADAPTERS)}", file=sys.stderr)
        return 2

    a = get_adapter(args.model_type)
    print(f"{a.display_name}  [{a.model_type}]")
    print(f"  KV strategy        : {a.kv_strategy}")
    print(f"  Stored KV compressed: {a.kv_compressed}")
    print(f"  Weights public     : {a.weights_public}")
    print(f"  Long context       : {a.supports_long_context}  (max {a.max_context or '-'})")
    print(f"  HF models          : {', '.join(a.hf_models) or '-'}")
    print(f"  Stand-in models    : {', '.join(a.standin_models) or '-'}")
    print(f"  Testable today     : {a.testable_model or '-'}")
    if a.notes:
        print(f"  Notes              : {a.notes}")

    # Worked analytical density example (typical GQA-ish config).
    example = analytical_density_kb_per_token_per_layer(
        a.model_type, num_key_value_heads=8, head_dim=128, kv_lora_rank=512,
    )
    if example is not None:
        print(f"  Analytical density : {example:.3f} KB/token/layer "
              f"(example: 8 KV heads, head_dim 128, latent 512, bf16)")
    return 0


def _cmd_api(args: argparse.Namespace) -> int:
    try:
        from src.api.cli import run as api_run
    except Exception as e:  # pragma: no cover - module may not be installed yet
        print(f"[kvscope api] API profiler unavailable: {e}", file=sys.stderr)
        print("Install API extras and ensure src/api is present.", file=sys.stderr)
        return 1
    return api_run(args.api_args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kvscope", description="KVScope: cross-architecture KV-cache profiler")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list known architectures and metadata")

    info = sub.add_parser("info", help="show details for one architecture")
    info.add_argument("model_type", help="e.g. minimax_m3, deepseek_v4, gemma4")

    api = sub.add_parser("api", help="run the black-box API behavioural profiler")
    api.add_argument("api_args", nargs=argparse.REMAINDER,
                     help="arguments forwarded to the API profiler")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "info":
        return _cmd_info(args)
    if args.command == "api":
        return _cmd_api(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
