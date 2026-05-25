"""
mha_baseline_runner.py — Pure multi-head-attention KV cache baseline
======================================================================

A small, intentionally classical model that uses *vanilla* multi-head
attention (one KV head per Q head, no GQA, no MQA, no MLA, no sliding
window, no MoE). This is the architectural anchor against which the
memory savings of GQA / GQA+sliding / MLA / hybrid stacks are measured
in the paper.

Default model: ``EleutherAI/pythia-1.4b-deduped``
    24 layers · 16 heads · head_dim=128 · max_position=2048 · pure MHA
    Per-token-per-layer KV (bf16) = 16 · 128 · 2 · 2 = 8192 bytes ≈ 8 KB.

For the paper this number anchors the cross-architecture density bar
chart: every other model's lower number is *exactly* the GQA / MQA /
MLA savings expressed as bytes per token per layer.

Usage:
    python -m src.models.mha_baseline_runner \\
        --model EleutherAI/pythia-1.4b-deduped \\
        --max-tokens 1024 \\
        --output ./results/mha_baseline_profile.json \\
        --prompts-file ./results/prompts.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Compatibility shims for any trust_remote_code / DynamicCache quirks
from src.profiler import compat  # noqa: F401
from src.profiler.json_utils import dump_json
from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector


def run(model_id: str, prompts: List[str], max_new_tokens: int,
        output_path: str, capture_every_n_steps: int = 1,
        device: str = "cuda") -> None:
    print(f"[mha-baseline] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    cfg = model.config
    print(f"[mha-baseline] Model loaded.")
    print(f"  num_hidden_layers : {getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layer', '?'))}")
    print(f"  num_attention_heads: {getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_head', '?'))}")
    n_kv = getattr(cfg, "num_key_value_heads", None)
    n_q = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if n_kv is None or n_kv == n_q:
        print(f"  KV heads          : {n_q} (== Q heads → pure MHA ✓)")
    else:
        print(f"  WARNING: num_key_value_heads={n_kv} != num_attention_heads={n_q}; "
              f"this is GQA, not MHA. Choose a different model for the baseline.")
    print(f"  hidden_size       : {getattr(cfg, 'hidden_size', '?')}")
    print(f"  max_position      : {getattr(cfg, 'max_position_embeddings', '?')}")

    nvml = NVMLSampler()
    nvml.set_baseline()

    all_reports = []
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[mha-baseline] Prompt {idx}/{len(prompts)}")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=getattr(cfg, "max_position_embeddings", 2048))
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        baseline = nvml.sample()["used_mb"]

        tracer = KVCacheTracer(
            model, model_type="mha",
            capture_every_n_steps=capture_every_n_steps,
            verbose=True,
        )
        with tracer:
            t0 = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_time = time.time() - t0
        actual_new_tokens = output.shape[1] - input_len
        peak = nvml.sample()["used_mb"]

        # Free the cache (mirrors what other runners do)
        del output
        torch.cuda.empty_cache()
        gc.collect()
        post_eos = nvml.sample()["used_mb"]

        report = tracer.report()
        df = tracer.snapshots_as_dataframe()
        detector = KVLeakDetector("mha")
        leak = detector.analyze(df, baseline, peak, post_eos)
        detector.print_report(leak)

        all_reports.append({
            "prompt_idx": idx,
            "input_tokens": input_len,
            "actual_new_tokens": actual_new_tokens,
            "generation_time_s": round(gen_time, 2),
            "tokens_per_sec": round(actual_new_tokens / gen_time, 1) if gen_time > 0 else 0,
            "tracer": report,
            "leak_score": leak.overall_leak_score,
            "leak_detection": {
                "overall_score": leak.overall_leak_score,
                "summary": leak.summary,
                "findings": [
                    {"detector": f.detector, "severity": f.severity,
                     "score": f.score, "description": f.description,
                     "evidence": getattr(f, "evidence", {}) or {}}
                    for f in leak.findings
                ],
            },
            "memory": {
                "baseline_mb": round(baseline, 2),
                "peak_mb": round(peak, 2),
                "post_eos_mb": round(post_eos, 2),
                "overhead_mb": round(peak - baseline, 2),
            },
        })

    final = {
        "model_type": "mha",
        "model_id": model_id,
        "per_prompt": all_reports,
        "aggregate": {
            "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in all_reports)
                                  / max(1, len(all_reports)),
            "avg_peak_overhead_mb": sum(r["memory"]["overhead_mb"]
                                        for r in all_reports)
                                    / max(1, len(all_reports)),
            "max_leak_score": max(r["leak_score"] for r in all_reports),
        },
        "architecture_role": (
            "Pure multi-head attention baseline. Per-token-per-layer KV "
            "= num_attention_heads * head_dim * 2 (K+V) * dtype_size. "
            "GQA / MQA / MLA models save bytes by reducing the effective "
            "head count or by storing a compressed latent."
        ),
    }
    dump_json(final, output_path)
    print(f"[mha-baseline] Done. Wrote {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b-deduped",
                    help="HF model id (must be pure MHA: num_kv_heads == num_q_heads)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompts-file", required=True,
                    help="Path to a JSON array of prompt strings")
    ap.add_argument("--capture-every", type=int, default=1)
    args = ap.parse_args()

    with open(args.prompts_file) as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        print(f"[mha-baseline] Bad prompts file: expected JSON array of strings",
              file=sys.stderr)
        sys.exit(2)

    run(args.model, prompts, args.max_tokens, args.output,
        capture_every_n_steps=args.capture_every)


if __name__ == "__main__":
    main()
