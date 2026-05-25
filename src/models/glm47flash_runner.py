"""
glm47flash_runner.py — GLM-4.7-Flash Profiled Runner
==========================================================
GLM-4.7-Flash uses a MoE architecture
with DeepSeek Sparse Attention (DSA). Key characteristics:

  - 744B total / 40B active parameters (GLM-5.1 full)
  - 30B MoE (GLM-4.7-Flash — our runnable proxy)
  - DSA reduces long-context compute by sparsifying attention
  - KV cache behavior is standard GQA but expert routing affects
    which FFN experts fire, creating non-uniform compute load

For our profiling purposes, GLM-4.7-Flash is architecturally representative
of the GLM family's KV cache behavior. The attention mechanism is the same;
only the FFN routing differs between model sizes.

This runner is a thin wrapper around the generic profiling infrastructure.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Compatibility shim for trust_remote_code modeling files
from src.profiler import compat  # noqa: F401

from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector
from src.profiler.json_utils import dump_json


class GLM47FlashRunner:
    """Profiled inference runner for GLM-4.7-Flash."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        load_in_8bit: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GLM47FlashRunner] VRAM: {vram:.1f}GB")

        if load_in_8bit:
            self.quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            self.quant_config = None

    def load(self):
        print(f"[GLM47FlashRunner] Loading {self.model_path}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if self.quant_config is not None:
            load_kwargs["quantization_config"] = self.quant_config
            load_kwargs["device_map"] = "auto"
        else:
            # Avoid Float8 conversion ops on CUDA during load; place on CPU first.
            load_kwargs["device_map"] = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        if self.quant_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[GLM47FlashRunner] Loaded in {time.time()-t0:.1f}s")

        # Print MoE config if available
        config = self.model.config
        for attr in ["num_experts", "num_experts_per_tok", "num_attention_heads",
                     "num_key_value_heads", "hidden_size", "num_hidden_layers"]:
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
            print(f"\n[GLM47FlashRunner] Prompt {idx+1}/{len(prompts)}")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # GLM checkpoints may emit token_type_ids that causal LM forward does not accept.
            inputs.pop("token_type_ids", None)
            input_len = inputs["input_ids"].shape[1]

            baseline = nvml.sample()["used_mb"]
            nvml.set_baseline()

            tracer = KVCacheTracer(
                self.model, model_type="glm47flash",
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
                detector = KVLeakDetector("glm47flash")
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
                print(f"[GLM47FlashRunner] ERROR in leak_detector: {e}")
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
            "model_type": "glm47flash",
            "model_path": self.model_path,
            "per_prompt": all_reports,
            "aggregate": {
                "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in all_reports) / len(all_reports),
                "avg_peak_mb": sum(r["memory"]["overhead_mb"] for r in all_reports) / len(all_reports),
            },
        }

        if save_report_to:
            dump_json(final, save_report_to)
            print(f"\n[GLM47FlashRunner] Report saved to {save_report_to}")

        return final



def main():
    parser = argparse.ArgumentParser(description="Profile GLM-4.7-Flash KV cache behavior")
    parser.add_argument("--model-path", type=str, required=True, help="Path to GLM model")
    parser.add_argument("--prompts-file", type=str, required=True, help="Path to prompts JSON")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--capture-every", type=int, default=5, help="Capture every N steps")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")

    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts = json.load(f)

    runner = GLM47FlashRunner(args.model_path, device=args.device, load_in_8bit=args.load_in_8bit)
    runner.load()
    runner.run_profiled(
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        capture_every_n=args.capture_every,
        save_report_to=args.output,
    )


if __name__ == "__main__":
    main()
