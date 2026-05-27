"""
gemma4_runner.py: Gemma 4 Profiled Inference Runner

Gemma 4 KV cache architecture:
  - GQA: 2 Q heads share 1 KV head (local layers) / 8 Q heads share 1 KV head (global)
  - Shared KV Cache: last N layers reuse KV from earlier layers
  - Sliding window: local layers attend to only 512/1024 tokens
  - Global layers: attend to full context (up to 256K)
  - K == V in global attention layers (unique constraint; cache is halved)

Supported models:
  gemma-4-2b-it:  2B dense, fits fp16 on T4 (16GB)
  gemma-4-9b-it:  9B dense, fits Q8 on L4 (24GB) or fp16 on A100
  gemma-4-27b-it: 27B dense, fits Q4 on A100 40GB
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector
from src.profiler.json_utils import dump_json


# ─── Quantization Configs ─────────────────────────────────────────────────────

def get_quant_config(vram_gb: int) -> Optional[BitsAndBytesConfig]:
    """Select quantization based on available VRAM."""
    if vram_gb >= 40:
        return None  # Full fp16 on A100 40GB
    elif vram_gb >= 24:
        # 8-bit: ~2x compression, minimal quality loss
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        # 4-bit: fits 9B on 16GB, some quality loss
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )


def detect_vram_gb(device_idx: int = 0) -> int:
    """Returns available VRAM in GB."""
    if not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(device_idx)
    return int(props.total_memory / 1e9)


# ─── Gemma 4 Runner ───────────────────────────────────────────────────────────

class Gemma4Runner:
    """
    Loads Gemma 4 and runs profiled generation.
    All KV cache metrics are collected via KVCacheTracer hooks.
    """

    # Maps model_id → expected local/global layer pattern
    LAYER_PATTERNS = {
        "gemma-4-E2B": {"total": 18, "global_every_n": 4},
        "gemma-4-E4B": {"total": 32, "global_every_n": 4},
        "gemma-4-26B": {"total": 48, "global_every_n": 4},
        "gemma-4-31B": {"total": 52, "global_every_n": 4},
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        force_quantization: Optional[str] = None,  # "4bit" | "8bit" | None
        attn_implementation: str = "eager",  # "eager" | "sdpa" | "flash_attention_2"
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.attn_implementation = attn_implementation

        self.vram_gb = detect_vram_gb()
        print(f"[Gemma4Runner] VRAM: {self.vram_gb}GB | Device: {device}")

        if force_quantization == "4bit":
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif force_quantization == "8bit":
            self.quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            self.quant_config = get_quant_config(self.vram_gb)

        quant_str = "fp16" if self.quant_config is None else (
            "int8" if self.quant_config.load_in_8bit else "nf4"
        )
        print(f"[Gemma4Runner] Quantization: {quant_str}")

    def load(self):
        """Load model and tokenizer."""
        print(f"[Gemma4Runner] Loading {self.model_path}...")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Gemma 4 uses trust_remote_code for custom attention implementations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=self.quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.quant_config is None else None,
            attn_implementation=self.attn_implementation,
            trust_remote_code=True,
        )
        self.model.eval()

        load_time = time.time() - t0
        print(f"[Gemma4Runner] Loaded in {load_time:.1f}s")
        print(f"[Gemma4Runner] Model params: {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B")

        # Print layer structure for verification
        self._print_attention_layers()

    def _print_attention_layers(self):
        """Show the local/global attention interleaving pattern."""
        print("[Gemma4Runner] Attention layer structure:")
        seen = 0
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and "layers" in name:
                parts = [p for p in name.split(".") if p.isdigit()]
                if parts and seen < 12:  # Show first 12 layers
                    idx = int(parts[0])
                    is_global = (idx + 1) % 4 == 0
                    window = getattr(module, "sliding_window", None)
                    print(f"    Layer {idx:2d}: {'GLOBAL' if is_global else 'local ':6s} "
                          f"| window={window}")
                    seen += 1
        if seen > 0:
            print(f"    ... (showing {seen} of {seen}+more layers)")

    def run_profiled(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        capture_every_n: int = 5,
        save_report_to: Optional[str] = None,
    ) -> Dict:
        """
        Run generation with full KV cache profiling.

        Args:
            prompts: List of input strings
            max_new_tokens: Max tokens to generate per prompt
            capture_every_n: Capture KV snapshot every N decode steps (saves memory)
            save_report_to: Path to save JSON report

        Returns:
            Full profiling report dict
        """
        assert self.model is not None, "Call .load() first"

        nvml = NVMLSampler()
        baseline_mb = nvml.sample()["used_mb"]

        all_reports = []

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n[Gemma4Runner] Prompt {prompt_idx+1}/{len(prompts)}: {prompt[:60]}...")

            # Apply chat template to prevent immediate EOS on instruction-tuned models
            try:
                if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Manual Gemma chat format
                    formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            except Exception:
                formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

            # Tokenize
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            input_len = inputs["input_ids"].shape[1]
            print(f"[Gemma4Runner] Input length: {input_len} tokens")

            # Profile
            tracer = KVCacheTracer(
                self.model,
                model_type="gemma4",
                capture_every_n_steps=capture_every_n,
                verbose=True,
            )
            nvml.set_baseline()

            with tracer:
                t0 = time.time()

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )

                gen_time = time.time() - t0

            # Count actual generated tokens before deleting outputs
            if hasattr(outputs, 'sequences'):
                actual_new_tokens = outputs.sequences.shape[1] - input_len
            else:
                actual_new_tokens = outputs.shape[1] - input_len

            peak_mb = nvml.sample()["used_mb"]

            # Force cache release
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()

            post_eos_mb = nvml.sample()["used_mb"]

            # Collect report
            tracer_report = tracer.report()
            df = tracer.snapshots_as_dataframe()

            # Leak detection
            detector = KVLeakDetector("gemma4")
            leak_report = detector.analyze(df, baseline_mb, peak_mb, post_eos_mb)
            detector.print_report(leak_report)

            # Per-step timing
            tokens_per_sec = actual_new_tokens / gen_time if gen_time > 0 else 0

            prompt_report = {
                "prompt_idx": prompt_idx,
                "prompt_preview": prompt[:100],
                "input_tokens": input_len,
                "actual_new_tokens": actual_new_tokens,
                "model": self.model_path,
                "generation_time_s": round(gen_time, 2),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "tracer": tracer_report,
                "leak_detection": {
                    "overall_score": leak_report.overall_leak_score,
                    "has_leaks": leak_report.has_leaks,
                    "summary": leak_report.summary,
                    "findings": [
                        {
                            "detector": f.detector,
                            "severity": f.severity,
                            "score": f.score,
                            "description": f.description,
                            "evidence": getattr(f, "evidence", {}) or {},
                        }
                        for f in leak_report.findings
                    ],
                },
                "memory": {
                    "baseline_mb": round(baseline_mb, 2),
                    "peak_mb": round(peak_mb, 2),
                    "post_eos_mb": round(post_eos_mb, 2),
                    "overhead_mb": round(peak_mb - baseline_mb, 2),
                    "unreleased_mb": round(post_eos_mb - baseline_mb, 2),
                },
            }
            all_reports.append(prompt_report)

        # Aggregate
        final_report = {
            "model_type": "gemma4",
            "model_path": self.model_path,
            "vram_gb": self.vram_gb,
            "n_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
            "per_prompt": all_reports,
            "aggregate": {
                "avg_tokens_per_sec": sum(r["tokens_per_sec"] for r in all_reports) / len(all_reports),
                "avg_peak_overhead_mb": sum(r["memory"]["overhead_mb"] for r in all_reports) / len(all_reports),
                "max_leak_score": max(r["leak_detection"]["overall_score"] for r in all_reports),
            },
        }

        if save_report_to:
            dump_json(final_report, save_report_to)
            print(f"\n[Gemma4Runner] Report saved to {save_report_to}")

        return final_report


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 KV Cache Profiler")
    parser.add_argument("--model", default="~/models/gemma4", help="Model path")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="none")
    parser.add_argument("--output", default="results/gemma4_profile.json")
    parser.add_argument("--prompts-file", default=None, help="JSON file with prompt list")
    args = parser.parse_args()

    DEFAULT_PROMPTS = [
        "Explain the transformer attention mechanism in detail, covering self-attention, multi-head attention, and the role of queries, keys, and values.",
        "Write a detailed analysis of the economic factors that led to the 2008 financial crisis, including the role of mortgage-backed securities and credit default swaps.",
        "Describe the process of protein folding and why it's important for drug discovery. Include discussion of AlphaFold and its impact.",
    ]

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS

    runner = Gemma4Runner(
        model_path=args.model,
        force_quantization=None if args.quant == "none" else args.quant,
    )
    runner.load()
    report = runner.run_profiled(
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        save_report_to=args.output,
    )

    print(f"\n✅ Done. Avg throughput: {report['aggregate']['avg_tokens_per_sec']:.1f} tok/s")
    print(f"   Avg KV overhead: {report['aggregate']['avg_peak_overhead_mb']:.1f} MB")
    print(f"   Max leak score: {report['aggregate']['max_leak_score']:.3f}")
