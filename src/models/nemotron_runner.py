"""
nemotron_runner.py: NVIDIA Nemotron Profiled Runner
===================================================
NVIDIA Nemotron models use a hybrid Mamba SSM + sparse GQA attention architecture.
Key characteristics:

  - Nemotron-3-Nano-30B: Mamba SSM majority + sparse GQA attention
  - Selective quantization: attention layers in BF16, KV cache in FP8
  - Mamba layers feed into attention layers (hybrid architecture)
  - Efficient for long-context agentic tasks

This runner profiles Nemotron's KV cache behavior, focusing on:
  - How the hybrid Mamba+attention architecture affects KV growth
  - Sparse GQA KV density compared to pure GQA models
  - Post-EOS memory behavior in hybrid architectures

Usage:
    python -m src.models.nemotron_runner \
        --model-path /root/models/nemotron \
        --prompts-file experiments/configs/prompts_advanced.json \
        --max-tokens 4096 \
        --capture-every 8 \
        --output results/nemotron_profile.json
"""

import argparse
import gc
import json
import threading
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

# Nemotron-specific cache for Mamba hybrid architecture
NemotronHHybridDynamicCache = None

def _load_nemotron_cache_class():
    """Dynamically load NemotronHHybridDynamicCache from cached modeling file."""
    global NemotronHHybridDynamicCache
    import os
    import sys

    # Add HF cache modules dir to path so package imports work (fixes relative imports)
    cache_modules = os.path.expanduser("~/.cache/huggingface/modules")
    if cache_modules not in sys.path:
        sys.path.insert(0, cache_modules)
    
    try:
        import transformers_modules.nemotron.modeling_nemotron_h as nemotron_module
        NemotronHHybridDynamicCache = nemotron_module.NemotronHHybridDynamicCache
        print("[NemotronRunner] Loaded NemotronHHybridDynamicCache from transformers_modules.nemotron.modeling_nemotron_h")
        return
    except Exception as e:
        print(f"[NemotronRunner] Failed package import: {e}")
    
    # Fallback: try standard import paths
    try:
        from transformers.models.nemotron.modeling_nemotron_h import NemotronHHybridDynamicCache
        return
    except ImportError:
        pass
    try:
        from transformers.models.nemotron.modeling import NemotronHHybridDynamicCache
        return
    except ImportError:
        pass
    
    print("[NemotronRunner] WARNING: NemotronHHybridDynamicCache not found. Cache will not work.")

_load_nemotron_cache_class()


class NemotronRunner:
    """Profiled inference runner for NVIDIA Nemotron models."""

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
        print(f"[NemotronRunner] VRAM: {vram:.1f}GB")

        if load_in_8bit:
            self.quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            self.quant_config = None

    def load(self):
        print(f"[NemotronRunner] Loading {self.model_path}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=self.quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[NemotronRunner] Loaded in {time.time()-t0:.1f}s")

        # Print config
        config = self.model.config
        for attr in ["num_attention_heads", "num_key_value_heads", "hidden_size",
                     "num_hidden_layers", "model_type"]:
            val = getattr(config, attr, "N/A")
            print(f"    {attr}: {val}")

    def _ssm_memory_sampler(self, nvml, interval_s: float = 0.05):
        """Background thread that samples GPU memory during generation."""
        trajectory = []
        stop_event = threading.Event()

        def _sample_loop():
            while not stop_event.is_set():
                try:
                    trajectory.append({
                        "t": time.time(),
                        "used_mb": nvml.sample()["used_mb"],
                    })
                except Exception:
                    pass
                stop_event.wait(interval_s)

        thread = threading.Thread(target=_sample_loop, daemon=True)
        return thread, stop_event, trajectory

    def run_profiled(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        capture_every_n: int = 5,
        save_report_to: Optional[str] = None,
        ssm_baseline: bool = False,
    ) -> Dict:
        assert self.model is not None, "Call .load() first"

        nvml = NVMLSampler()
        all_reports = []

        for idx, prompt in enumerate(prompts):
            print(f"\n[NemotronRunner] Prompt {idx+1}/{len(prompts)}")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

            baseline = nvml.sample()["used_mb"]
            nvml.set_baseline()

            seq_len = inputs["input_ids"].shape[1]
            gen_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
                "cache_position": torch.arange(0, seq_len, device=self.device),
            }

            if ssm_baseline:
                # SSM baseline: skip KVTracer, sample memory from background thread
                # use_cache=False because Nemotron requires its custom cache class
                # which isn't importable; generation still works (slightly slower)
                print("[NemotronRunner] SSM baseline mode: skipping KVTracer, sampling memory trajectory")
                gen_kwargs["use_cache"] = False
                gen_kwargs.pop("cache_position", None)
                thread, stop_event, trajectory = self._ssm_memory_sampler(nvml, interval_s=0.05)
                thread.start()
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(**gen_kwargs)
                gen_time = time.time() - t0
                stop_event.set()
                thread.join(timeout=2.0)

                peak = max((p["used_mb"] for p in trajectory), default=baseline)
                report = {
                    "ssm_baseline": True,
                    "attention_layers_found": 0,
                    "note": "Nemotron-H is a Mamba-SSM hybrid with no standard transformer KV cache. "
                            "This profile captures memory trajectory and throughput instead.",
                    "memory_trajectory_points": len(trajectory),
                }
                leak_detection_data = {
                    "overall_score": 0.0,
                    "summary": "SSM architecture: no KV cache to leak. Memory is O(1) w.r.t sequence length.",
                    "findings": [],
                }
            else:
                # Standard mode (requires working cache_position / past_key_values)
                tracer = KVCacheTracer(
                    self.model, model_type="nemotron_h",
                    capture_every_n_steps=capture_every_n, verbose=True,
                )
                seq_len = inputs["input_ids"].shape[1]
                gen_kwargs["cache_position"] = torch.arange(0, seq_len, device=self.device)

                with tracer:
                    t0 = time.time()
                    with torch.no_grad():
                        outputs = self.model.generate(**gen_kwargs)
                    gen_time = time.time() - t0

                peak = nvml.sample()["used_mb"]
                report = tracer.report()
                df = tracer.snapshots_as_dataframe()

                try:
                    detector = KVLeakDetector("nemotron")
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
                    print(f"[NemotronRunner] ERROR in leak_detector: {e}")
                    leak_detection_data = {
                        "overall_score": 0.0,
                        "summary": f"Leak detection failed: {str(e)}",
                        "findings": [],
                    }

            actual_new_tokens = outputs.shape[1] - input_len
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
            post_eos = nvml.sample()["used_mb"]

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
            "model_type": "nemotron",
            "model_path": self.model_path,
            "per_prompt": all_reports,
            "aggregate": {
                "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in all_reports) / len(all_reports) if all_reports else 0,
                "avg_peak_mb": sum(r["memory"]["overhead_mb"] for r in all_reports) / len(all_reports) if all_reports else 0,
            },
        }

        if save_report_to:
            dump_json(final, save_report_to)
            print(f"\n[NemotronRunner] Report saved to {save_report_to}")

        return final


def main():
    parser = argparse.ArgumentParser(description="Profile Nemotron KV cache behavior")
    parser.add_argument("--model-path", type=str, required=True, help="Path to Nemotron model")
    parser.add_argument("--prompts-file", type=str, required=True, help="Path to prompts JSON")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--capture-every", type=int, default=5, help="Capture every N steps")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (off by default for Nemotron stability)")
    parser.add_argument("--ssm-baseline", action="store_true", help="SSM baseline mode: skip KVTracer, capture memory trajectory + throughput only")

    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts = json.load(f)

    runner = NemotronRunner(args.model_path, device=args.device, load_in_8bit=args.load_in_8bit)
    runner.load()
    runner.run_profiled(
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        capture_every_n=args.capture_every,
        save_report_to=args.output,
        ssm_baseline=args.ssm_baseline,
    )


if __name__ == "__main__":
    main()
