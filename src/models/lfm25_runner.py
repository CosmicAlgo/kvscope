"""
lfm25_runner.py: LiquidAI LFM2.5-350M Profiled Runner
======================================================
LFM2.5-350M uses a hybrid Linear Input-Varying Systems (LIV) + GQA architecture.
Key characteristics:

  - 350M parameters (very small compared to other profiled models)
  - 16 layers: 10 double-gated LIV convolution blocks + 6 GQA blocks
  - Hybrid architecture: LIV convolution + Grouped Query Attention
  - Context length: 32,768 tokens
  - Departure from pure Transformer architecture

This runner profiles LFM2.5's KV cache behavior, focusing on:
  - How the hybrid LIV+GQA architecture affects KV growth
  - KV density in the 6 GQA blocks vs pure attention models
  - Comparison with pure attention architectures (MHA, GQA, MLA, DSA)

Note: Only the 6 GQA blocks will have traditional KV cache. The 10 LIV convolution
blocks use Linear Input-Varying Systems, which have different memory characteristics.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Compatibility shim for trust_remote_code modeling files
from src.profiler import compat  # noqa: F401

from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector
from src.profiler.json_utils import dump_json


class LFM25Runner:
    """Profiled inference runner for LiquidAI LFM2.5-350M."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[LFM25Runner] VRAM: {vram:.1f}GB")

    def load(self):
        print(f"[LFM25Runner] Loading {self.model_path}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[LFM25Runner] Loaded in {time.time()-t0:.1f}s")

        # Print config
        config = self.model.config
        for attr in ["num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                     "hidden_size", "model_type", "vocab_size"]:
            val = getattr(config, attr, "N/A")
            print(f"    {attr}: {val}")

    def run_profiled(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        capture_every_n: int = 5,
        save_report_to: Optional[str] = None,
    ) -> Dict:
        assert self.model is not None, "Call .load() first"

        nvml = NVMLSampler()
        all_reports = []

        for idx, prompt in enumerate(prompts):
            print(f"\n[LFM25Runner] Prompt {idx+1}/{len(prompts)}")

            # Apply chat template to prevent immediate EOS on instruction-tuned models
            try:
                if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Prefix to encourage generation for base models
                    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
            except Exception:
                formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

            baseline = nvml.sample()["used_mb"]
            nvml.set_baseline()

            tracer = KVCacheTracer(
                self.model, model_type="lfm25",
                capture_every_n_steps=capture_every_n, verbose=True,
            )
            with tracer:
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, use_cache=True,
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
                detector = KVLeakDetector("lfm25")
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
                print(f"[LFM25Runner] ERROR in leak_detector: {e}")
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

        final = {
            "model_type": "lfm25",
            "model_path": self.model_path,
            "per_prompt": all_reports,
            "aggregate": {
                "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in all_reports) / len(all_reports),
                "avg_peak_mb": sum(r["memory"]["overhead_mb"] for r in all_reports) / len(all_reports),
            },
        }

        if save_report_to:
            dump_json(final, save_report_to)
            print(f"\n[LFM25Runner] Report saved to {save_report_to}")

        return final


def main():
    parser = argparse.ArgumentParser(description="Profile LFM2.5-350M KV cache behavior")
    parser.add_argument("--model-path", type=str, required=True, help="Path to LFM2.5-350M model")
    parser.add_argument("--prompts-file", type=str, required=True, help="Path to prompts JSON")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--capture-every", type=int, default=5, help="Capture every N steps")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    # Load prompts
    with open(args.prompts_file) as f:
        prompts = json.load(f)
    
    # Run profiling
    runner = LFM25Runner(args.model_path, device=args.device)
    runner.load()
    runner.run_profiled(
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        capture_every_n=args.capture_every,
        save_report_to=args.output,
    )


if __name__ == "__main__":
    main()
