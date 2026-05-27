"""
qwen36_runner.py: Qwen3.6 Profiled Runner
===================================================
Qwen3.6 has a hybrid architecture:
- 75% Gated DeltaNet layers (no traditional KV cache - uses state-space-like linear attention)
- 25% Gated Attention layers (GQA with 2 KV heads)
- 40 layers total in 10 blocks of 4 layers each (35B-A3B variant)

This runner captures KV cache only for the Gated Attention layers.
DeltaNet layers have minimal KV-like state (gate parameters, not traditional KV).

Usage:
    python -m src.models.qwen36_runner \
        --model-path /root/models/qwen36 \
        --prompts-file experiments/configs/prompts_advanced.json \
        --max-tokens 300 \
        --capture-every 5 \
        --output results/qwen36_profile.json
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector
from src.profiler.json_utils import dump_json


class Qwen36Runner:
    """Profiled inference runner for Qwen3.6 models."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        torch_dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.torch_dtype = torch_dtype

        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Qwen36Runner] VRAM: {vram:.1f}GB")
        print(f"[Qwen36Runner] Note: Qwen3.6 uses hybrid DeltaNet + Gated Attention")
        print(f"[Qwen36Runner] Only Gated Attention layers (25%) have traditional KV cache")

    def load(self):
        print(f"[Qwen36Runner] Loading {self.model_path}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Qwen3.6-35B-A3B ~72GB in bf16, fits in 141GB H200
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        load_time = time.time() - t0
        print(f"[Qwen36Runner] Loaded in {load_time:.1f}s")

        config = self.model.config
        for attr in ["num_attention_heads", "num_key_value_heads", "hidden_size",
                     "num_hidden_layers", "model_type"]:
            val = getattr(config, attr, "N/A")
            print(f"    {attr}: {val}")

        # Print architecture summary
        self._print_architecture_summary()

    def _print_architecture_summary(self):
        """Show the DeltaNet vs Gated Attention layer distribution."""
        n_layers = getattr(self.model.config, "num_hidden_layers", 40)
        n_attn = n_layers // 4
        n_delta = n_layers - n_attn
        n_blocks = n_layers // 4
        print(f"[Qwen36Runner] Architecture: {n_blocks} blocks × 4 layers = {n_layers} total layers")
        print(f"[Qwen36Runner] Per block: 3× Gated DeltaNet + 1× Gated Attention")
        print(f"[Qwen36Runner] Total: {n_delta} DeltaNet layers (no KV) + {n_attn} Gated Attention layers (GQA)")
        print(f"[Qwen36Runner] Expected: ~75% KV cache reduction vs pure attention model")

    def run_profiled(
        self,
        prompts: List[str],
        max_new_tokens: int = 300,
        capture_every_n: int = 5,
        save_report_to: Optional[str] = None,
    ) -> Dict:
        assert self.model is not None, "Call .load() first"

        nvml = NVMLSampler()
        all_reports = []

        for idx, prompt in enumerate(prompts):
            print(f"\n[Qwen36Runner] Prompt {idx+1}/{len(prompts)}")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]
            print(f"[Qwen36Runner] Input: {input_len} tokens")

            baseline = nvml.sample()["used_mb"]
            nvml.set_baseline()

            # Use qwen36 model_type - tracer will classify layers
            print(f"[Qwen36Runner] Creating KVCacheTracer with verbose=True")
            tracer = KVCacheTracer(
                self.model, model_type="qwen36",
                capture_every_n_steps=capture_every_n, verbose=True,
            )
            print(f"[Qwen36Runner] Tracer created, registering hooks...")

            with tracer:
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
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

            # Add layer type statistics
            layer_stats = self._compute_layer_stats(report)

            try:
                detector = KVLeakDetector("qwen36")
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
                print(f"[Qwen36Runner] ERROR in leak_detector: {e}")
                leak_detection_data = {
                    "overall_score": 0.0,
                    "summary": f"Leak detection failed: {str(e)}",
                    "findings": [],
                }

            tokens_per_sec = actual_new_tokens / gen_time if gen_time > 0 else 0

            all_reports.append({
                "prompt_idx": idx,
                "prompt_preview": prompt[:80],
                "input_tokens": input_len,
                "actual_new_tokens": actual_new_tokens,
                "generation_time_s": round(gen_time, 2),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "tracer": report,
                "layer_statistics": layer_stats,
                "leak_detection": leak_detection_data,
                "memory": {
                    "baseline_mb": round(baseline, 2),
                    "peak_mb": round(peak, 2),
                    "post_eos_mb": round(post_eos, 2),
                    "overhead_mb": round(peak - baseline, 2),
                    "unreleased_mb": round(post_eos - baseline, 2),
                },
            })

        # Aggregate statistics
        avg_tok_per_sec = sum(r["tokens_per_sec"] for r in all_reports) / len(all_reports)
        avg_overhead = sum(r["memory"]["overhead_mb"] for r in all_reports) / len(all_reports)
        max_leak = max(r["leak_detection"]["overall_score"] for r in all_reports)

        config = self.model.config
        n_layers = getattr(config, "num_hidden_layers", 40)
        n_attn = n_layers // 4
        n_delta = n_layers - n_attn
        n_q_heads = getattr(config, "num_attention_heads", 16)
        n_kv_heads = getattr(config, "num_key_value_heads", 2)

        final_report = {
            "model_type": "qwen36",
            "model_path": self.model_path,
            "architecture": {
                "total_layers": n_layers,
                "deltanet_layers": n_delta,
                "gated_attention_layers": n_attn,
                "attention_heads": {"q": n_q_heads, "kv": n_kv_heads, "head_dim": 256},
                "deltanet_heads": {"v": 32, "qk": 16, "head_dim": 128},
            },
            "n_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
            "per_prompt": all_reports,
            "aggregate": {
                "avg_tokens_per_sec": round(avg_tok_per_sec, 1),
                "avg_peak_overhead_mb": round(avg_overhead, 1),
                "max_leak_score": round(max_leak, 3),
            },
        }

        if save_report_to:
            dump_json(final_report, save_report_to)
            print(f"\n[Qwen36Runner] Report saved to {save_report_to}")

        print(f"\n✅ Qwen3.6 Done. Avg throughput: {avg_tok_per_sec:.1f} tok/s")
        print(f"   Avg GPU overhead: {avg_overhead:.1f} MB")
        print(f"   Max leak score: {max_leak:.3f}")

        return final_report

    def _compute_layer_stats(self, report: Dict) -> Dict:
        """Compute statistics about layer types (DeltaNet vs Attention)."""
        per_layer = report.get("kv_cache", {}).get("per_layer_mb", {})
        
        # Qwen3.6: layers 0, 4, 8, ... (multiples of 4) are Gated Attention
        # Others are DeltaNet (minimal/no traditional KV)
        attention_layers = {}
        deltanet_layers = {}
        
        for layer_idx, size_mb in per_layer.items():
            layer_num = int(layer_idx) if isinstance(layer_idx, str) else layer_idx
            if layer_num % 4 == 3:  # Every 4th layer (indices 3, 7, 11, ...)
                attention_layers[layer_idx] = size_mb
            else:
                deltanet_layers[layer_idx] = size_mb
        
        return {
            "attention_layers_count": len(attention_layers),
            "deltanet_layers_count": len(deltanet_layers),
            "attention_layers_total_mb": round(sum(attention_layers.values()), 2),
            "deltanet_layers_total_mb": round(sum(deltanet_layers.values()), 2),
            "attention_avg_mb_per_layer": round(sum(attention_layers.values()) / len(attention_layers), 3) if attention_layers else 0,
            "deltanet_avg_mb_per_layer": round(sum(deltanet_layers.values()) / len(deltanet_layers), 3) if deltanet_layers else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6 KV Cache Profiler")
    parser.add_argument("--model-path", default="/root/models/qwen36")
    parser.add_argument("--prompts-file", default="experiments/configs/prompts_advanced.json")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--capture-every", type=int, default=5)
    parser.add_argument("--output", default="results/qwen36_profile.json")
    args = parser.parse_args()

    prompts_file = Path(args.prompts_file)
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = [
            "Explain transformer attention mechanisms in detail.",
            "Write a comprehensive analysis of neural network architectures.",
            "Describe the process of protein folding and its importance.",
        ]

    runner = Qwen36Runner(args.model_path)
    runner.load()
    runner.run_profiled(
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        capture_every_n=args.capture_every,
        save_report_to=args.output,
    )


if __name__ == "__main__":
    main()
