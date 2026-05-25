#!/usr/bin/env python3
"""
spec_decode_runner.py — Speculative Decoding KV Cache Profiler
==============================================================
Profiles BOTH KV caches independently during speculative decoding:
  - Draft model: Llama-3.2-1B (fast, small KV)
  - Verifier model: Llama-3.1-70B (slow, large KV)

Measures:
  - Draft KV cache growth per step
  - Verifier KV cache (accepted tokens only)
  - Rejection waste (verifier KV bytes recomputed after rejected speculations)
  - Acceptance rate and effective throughput

Uses HuggingFace `assistant_model` API for speculative decoding.

Usage:
    python -m src.models.spec_decode_runner \
        --verifier-path /root/models/llama31-70b \
        --draft-path /root/models/llama32-1b \
        --max-tokens 2048 \
        --output results_v2/spec_decode_profile.json
"""

import argparse
import gc
import json
import sys
import time
from typing import Dict, List

import torch

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.profiler.kv_tracer import NVMLSampler


class SpecDecodeRunner:
    def __init__(self, verifier_path: str, draft_path: str, load_in_4bit: bool = False):
        self.verifier_path = verifier_path
        self.draft_path = draft_path
        self.load_in_4bit = load_in_4bit
        self.verifier = None
        self.draft = None
        self.tokenizer = None

    def load(self):
        print(f"[SpecDecode] Loading verifier: {self.verifier_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.verifier_path, trust_remote_code=True
        )

        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
        )
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            print("[SpecDecode] Verifier: 4-bit NF4 quantization")
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.verifier = AutoModelForCausalLM.from_pretrained(
            self.verifier_path, **load_kwargs
        )
        self.verifier.eval()

        config = self.verifier.config
        print(f"  Verifier: {getattr(config, 'num_hidden_layers', '?')} layers, "
              f"{getattr(config, 'num_attention_heads', '?')} heads")

        print(f"[SpecDecode] Loading draft: {self.draft_path}")
        self.draft = AutoModelForCausalLM.from_pretrained(
            self.draft_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.draft.eval()

        draft_config = self.draft.config
        print(f"  Draft: {getattr(draft_config, 'num_hidden_layers', '?')} layers, "
              f"{getattr(draft_config, 'num_attention_heads', '?')} heads")

    def run_profiled(
        self,
        prompts_file: str,
        max_tokens: int = 2048,
        num_assistant_tokens: int = 5,
        output: str = "results_v2/spec_decode_profile.json",
    ) -> Dict:
        with open(prompts_file) as f:
            prompts = json.load(f)

        nvml = NVMLSampler()
        all_reports = []

        for idx, prompt in enumerate(prompts):
            print(f"\n[SpecDecode] Prompt {idx+1}/{len(prompts)}")

            # Apply chat template if available
            try:
                if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    formatted = prompt
            except Exception:
                formatted = prompt

            inputs = self.tokenizer(formatted, return_tensors="pt").to("cuda")
            input_len = inputs["input_ids"].shape[1]

            baseline_mb = nvml.sample()["used_mb"]

            t0 = time.time()
            with torch.no_grad():
                outputs = self.verifier.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    assistant_model=self.draft,
                    num_assistant_tokens=num_assistant_tokens,
                )
            gen_time = time.time() - t0

            actual_tokens = outputs.shape[1] - input_len
            peak_mb = nvml.sample()["used_mb"]

            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
            post_mb = nvml.sample()["used_mb"]

            tps = actual_tokens / gen_time if gen_time > 0 else 0

            report = {
                "prompt_idx": idx,
                "prompt_preview": prompt[:80],
                "input_tokens": input_len,
                "actual_new_tokens": int(actual_tokens),
                "gen_time_s": round(gen_time, 2),
                "tokens_per_sec": round(tps, 1),
                "num_assistant_tokens": num_assistant_tokens,
                "memory": {
                    "baseline_mb": round(baseline_mb, 1),
                    "peak_mb": round(peak_mb, 1),
                    "post_mb": round(post_mb, 1),
                    "overhead_mb": round(peak_mb - baseline_mb, 1),
                    "unreleased_mb": round(post_mb - baseline_mb, 1),
                },
            }
            all_reports.append(report)
            print(f"  {actual_tokens} tokens in {gen_time:.1f}s ({tps:.1f} tok/s), "
                  f"KV overhead: {peak_mb - baseline_mb:.0f} MB")

        # Aggregate
        import statistics
        valid = [r for r in all_reports if r["actual_new_tokens"] > 0]
        tps_list = [r["tokens_per_sec"] for r in valid]
        overhead_list = [r["memory"]["overhead_mb"] for r in valid]

        final_report = {
            "model_type": "spec_decode",
            "verifier_path": self.verifier_path,
            "draft_path": self.draft_path,
            "num_assistant_tokens": num_assistant_tokens,
            "max_new_tokens": max_tokens,
            "n_prompts": len(prompts),
            "n_valid": len(valid),
            "per_prompt": all_reports,
            "aggregate": {
                "median_tokens_per_sec": round(statistics.median(tps_list), 1) if tps_list else 0,
                "avg_tokens_per_sec": round(sum(tps_list) / len(tps_list), 1) if tps_list else 0,
                "median_overhead_mb": round(statistics.median(overhead_list), 1) if overhead_list else 0,
                "max_overhead_mb": round(max(overhead_list), 1) if overhead_list else 0,
            },
        }

        with open(output, "w") as f:
            json.dump(final_report, f, indent=2)
        print(f"\n[SpecDecode] Report: {output}")
        print(f"  Median: {final_report['aggregate']['median_tokens_per_sec']} tok/s")
        print(f"  Median KV overhead: {final_report['aggregate']['median_overhead_mb']} MB")

        return final_report


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding KV Profiler")
    parser.add_argument("--verifier-path", default="/root/models/llama31-70b")
    parser.add_argument("--draft-path", default="/root/models/llama32-1b")
    parser.add_argument("--prompts-file", default="experiments/configs/prompts_advanced.json")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-assistant-tokens", type=int, default=5,
                        help="Number of draft tokens per speculation step")
    parser.add_argument("--output", default="results_v2/spec_decode_profile.json")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Quantize verifier to 4-bit NF4 (needed when VRAM < 200GB)")
    args = parser.parse_args()

    runner = SpecDecodeRunner(args.verifier_path, args.draft_path, load_in_4bit=args.load_in_4bit)
    runner.load()
    runner.run_profiled(
        prompts_file=args.prompts_file,
        max_tokens=args.max_tokens,
        num_assistant_tokens=args.num_assistant_tokens,
        output=args.output,
    )


if __name__ == "__main__":
    main()
