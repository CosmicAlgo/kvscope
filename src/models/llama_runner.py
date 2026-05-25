#!/usr/bin/env python3
"""
Llama (Meta) KV Cache Profiling Runner

Supports Llama-3.x and Llama-3.1.x models with standard dense attention.
Pattern matches qwen3_runner.py for consistent KVLeakDetector API usage.
"""

import argparse
import gc
import json
import sys
import time
import torch
from typing import Dict, List, Any

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector


class LlamaRunner:
    def __init__(self, model_path: str, load_in_4bit: bool = False):
        self.model_path = model_path
        self.load_in_4bit = load_in_4bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load(self):
        print(f"[LlamaRunner] Loading {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        load_kwargs = dict(device_map="auto", trust_remote_code=True)
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            print("[LlamaRunner] Using 4-bit NF4 quantization")
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )

        config = self.model.config
        print(f"[LlamaRunner] Loaded")
        print(f"    num_attention_heads: {getattr(config, 'num_attention_heads', 'N/A')}")
        print(f"    num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
        print(f"    hidden_size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"    num_hidden_layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"    model_type: {config.model_type}")

    def run_profiled(
        self,
        prompts_file: str,
        max_tokens: int,
        capture_every: int,
        output: str,
    ) -> Dict:
        with open(prompts_file, "r") as f:
            prompts = json.load(f)

        nvml = NVMLSampler()
        all_reports = []

        for idx, prompt in enumerate(prompts):
            print(f"\n[LlamaRunner] Prompt {idx+1}/{len(prompts)}")

            # Llama-3.1-70B is a base model — use raw prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

            baseline = nvml.sample()["used_mb"]
            nvml.set_baseline()

            tracer = KVCacheTracer(
                self.model, model_type="llama",
                capture_every_n_steps=capture_every, verbose=True,
            )

            with tracer:
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                gen_time = time.time() - t0

            actual_new_tokens = outputs.shape[1] - input_len
            peak = nvml.sample()["used_mb"]
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
            post_eos = nvml.sample()["used_mb"]

            report = tracer.report()
            df = tracer.snapshots_as_dataframe()

            try:
                detector = KVLeakDetector("llama")
                leak = detector.analyze(df, baseline, peak, post_eos)
                detector.print_report(leak)
                leak_detection_data = {
                    "overall_score": leak.overall_leak_score,
                    "summary": leak.summary,
                    "findings": [
                        {"detector": f.detector, "severity": f.severity,
                         "score": f.score, "description": f.description,
                         "evidence": getattr(f, "evidence", {}) or {}}
                        for f in leak.findings
                    ],
                }
            except Exception as e:
                print(f"[LlamaRunner] ERROR in leak_detector: {e}")
                leak_detection_data = {
                    "overall_score": 0.0,
                    "summary": f"Leak detection failed: {str(e)}",
                    "findings": [],
                }

            all_reports.append({
                "prompt_idx": idx,
                "input_tokens": input_len,
                "actual_new_tokens": actual_new_tokens,
                "generation_time_s": round(gen_time, 2),
                "tokens_per_sec": round(actual_new_tokens / gen_time, 1) if gen_time > 0 else 0,
                "tracer": report,
                "leak_detection": leak_detection_data,
                "memory": {
                    "baseline_mb": round(baseline, 2),
                    "peak_mb": round(peak, 2),
                    "post_eos_mb": round(post_eos, 2),
                    "overhead_mb": round(peak - baseline, 2),
                },
            })

        # Environment snapshot for provenance
        env_snapshot = {
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "vram_mb": torch.cuda.get_device_properties(0).total_mem // (1024*1024) if torch.cuda.is_available() else 0,
            "torch_version": torch.__version__,
        }
        try:
            import transformers
            env_snapshot["transformers_version"] = transformers.__version__
        except Exception:
            pass

        payload = {
            "model_path": self.model_path,
            "model_type": "llama",
            "num_layers": self.model.config.num_hidden_layers,
            "d_model": self.model.config.hidden_size,
            "num_attention_heads": getattr(self.model.config, "num_attention_heads", None),
            "num_key_value_heads": getattr(self.model.config, "num_key_value_heads", None),
            "max_new_tokens": max_tokens,
            "capture_every": capture_every,
            "environment": env_snapshot,
            "results": all_reports,
        }

        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[LlamaRunner] Report saved to {output}")
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--capture-every", type=int, default=5)
    parser.add_argument("--output", default="results/llama_profile.json")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Quantize to 4-bit NF4 (needed for 70B on H200)")
    args = parser.parse_args()

    runner = LlamaRunner(args.model_path, load_in_4bit=args.load_in_4bit)
    runner.load()
    runner.run_profiled(
        prompts_file=args.prompts_file,
        max_tokens=args.max_tokens,
        capture_every=args.capture_every,
        output=args.output,
    )


if __name__ == "__main__":
    main()
