"""
minimax_m3_runner.py: MiniMax M3 (MSA) Profiled Runner
======================================================
MiniMax M3 introduces MSA (MiniMax Sparse Attention): the KV cache is partitioned
into blocks that are read once ("KV outer gather Q"), keeping K/V uncompressed
(full precision) while scaling context to 1M tokens. This makes M3 a new
KV-cache *strategy* row alongside MLA (DeepSeek) and CSA/HCA (DeepSeek V4).

STATUS (provisional): written in May 2026, when M3 open weights were not yet
public. The ``--standin`` flag lets it load MiniMax-M2 (same MoE family)
instead. Unit tests cover the helpers only; this runner has not been run on
real M2 or M3 weights, and no results from it are in this repository. The M3
layer classifier in kv_tracer.py guesses attribute names and must be checked
against the released modeling code.

Usage:
    # With the M2 stand-in:
    python -m src.models.minimax_m3_runner \
        --model-path /models/minimax_m2 --standin \
        --prompts-file experiments/configs/prompts_advanced.json \
        --max-tokens 256 --capture-every 8 \
        --output results/minimax_m3_standin_profile.json

    # With real M3 weights:
    python -m src.models.minimax_m3_runner \
        --model-path /models/minimax_m3 \
        --prompts-file experiments/configs/prompts_advanced.json \
        --max-tokens 256 --capture-every 8 --multi-gpu \
        --output results/minimax_m3_profile.json
"""

import argparse
import gc
import json
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Compatibility shim for trust_remote_code modeling files (must precede load)
from src.profiler import compat  # noqa: F401

from src.profiler.adapters import analytical_density_kb_per_token_per_layer, get_adapter
from src.profiler.json_utils import dump_json
from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector

MODEL_TYPE = "minimax_m3"


class MiniMaxM3Runner:
    """Profiled inference runner for MiniMax M3 (and the M2 stand-in)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        multi_gpu: Optional[bool] = None,
        is_standin: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.is_standin = is_standin
        self.model = None
        self.tokenizer = None
        self.adapter = get_adapter(MODEL_TYPE)

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # Default: aggregate NVML across all GPUs when more than one is visible
        # (M3 is a 200-400B MoE → tensor-parallel; single-device VRAM undercounts).
        self.multi_gpu = (n_gpus > 1) if multi_gpu is None else multi_gpu
        print(f"[M3Runner] visible GPUs: {n_gpus}  multi_gpu={self.multi_gpu}")

        if load_in_4bit:
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            self.quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            self.quant_config = None

        if not self.adapter.weights_public and not is_standin:
            print("[M3Runner] NOTE: M3 open weights are not public yet. If the "
                  "model-path is unavailable, re-run with --standin to use "
                  f"{self.adapter.testable_model}.")

    def load(self):
        print(f"[M3Runner] Loading {self.model_path} "
              f"({'M2 stand-in' if self.is_standin else 'M3'})...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=self.quant_config,
            device_map="auto",
            torch_dtype="auto" if self.quant_config is None else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print(f"[M3Runner] Loaded in {time.time()-t0:.1f}s  dtype={self.model.dtype}")
        self._print_and_capture_arch()

    def _print_and_capture_arch(self) -> Dict:
        cfg = self.model.config
        attrs = ["model_type", "num_hidden_layers", "num_attention_heads",
                 "num_key_value_heads", "hidden_size", "head_dim",
                 "num_local_experts", "num_experts_per_tok", "sliding_window",
                 "max_position_embeddings"]
        arch = {}
        for a in attrs:
            v = getattr(cfg, a, None)
            if v is not None:
                arch[a] = v
                print(f"    {a}: {v}")

        n_kv = getattr(cfg, "num_key_value_heads", None)
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            hs = getattr(cfg, "hidden_size", None)
            n_q = getattr(cfg, "num_attention_heads", None)
            head_dim = (hs // n_q) if (hs and n_q) else None
        analytical = analytical_density_kb_per_token_per_layer(
            MODEL_TYPE, num_key_value_heads=n_kv, head_dim=head_dim)
        arch["analytical_density_kb_per_tok_per_layer"] = analytical
        if analytical is not None:
            print(f"    [analytical] stored KV density: {analytical:.3f} KB/tok/layer "
                  f"(MSA keeps K/V uncompressed, so storage ~= GQA)")
        self._arch = arch
        return arch

    def run_profiled(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        capture_every_n: int = 8,
        save_report_to: Optional[str] = None,
    ) -> Dict:
        assert self.model is not None, "Call .load() first"

        nvml = NVMLSampler("all" if self.multi_gpu else 0)
        all_reports = []

        for idx, prompt in enumerate(prompts):
            print(f"\n[M3Runner] Prompt {idx+1}/{len(prompts)}")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            input_len = inputs["input_ids"].shape[1]

            baseline = nvml.sample()["used_mb"]
            nvml.set_baseline()

            tracer = KVCacheTracer(
                self.model, model_type=MODEL_TYPE,
                device_idx="all" if self.multi_gpu else 0,
                capture_every_n_steps=capture_every_n, verbose=(idx == 0),
            )
            with tracer:
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, use_cache=True)
                gen_time = time.time() - t0

            peak = nvml.sample()["used_mb"]
            report = tracer.report()
            df = tracer.snapshots_as_dataframe()

            actual_new_tokens = int(outputs.shape[1] - input_len)
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
            # Measure post-EOS AFTER freeing tensors (correct ordering).
            post_eos = nvml.sample()["used_mb"]

            try:
                detector = KVLeakDetector(MODEL_TYPE)
                leak = detector.analyze(df, baseline, peak, post_eos)
                detector.print_report(leak)
                leak_data = {
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
                print(f"[M3Runner] leak detector error: {e}")
                leak_data = {"overall_score": 0.0,
                             "summary": f"Leak detection failed: {e}", "findings": []}

            all_reports.append({
                "prompt_idx": idx,
                "input_tokens": input_len,
                "actual_new_tokens": actual_new_tokens,
                "generation_time_s": round(gen_time, 2),
                "tokens_per_sec": round(actual_new_tokens / gen_time, 1) if gen_time > 0 else 0,
                "tracer": report,
                "leak_detection": leak_data,
                "memory": {
                    "baseline_mb": round(baseline, 2),
                    "peak_mb": round(peak, 2),
                    "post_eos_mb": round(post_eos, 2),
                    "overhead_mb": round(peak - baseline, 2),
                },
            })

        final = {
            "model_type": MODEL_TYPE,
            "model_path": self.model_path,
            "is_standin": self.is_standin,
            "standin_model": self.model_path if self.is_standin else None,
            "kv_strategy": self.adapter.kv_strategy,
            "architecture": getattr(self, "_arch", {}),
            "per_prompt": all_reports,
            "aggregate": {
                "avg_tokens_per_sec": _avg(r["tokens_per_sec"] for r in all_reports),
                "avg_overhead_mb": _avg(r["memory"]["overhead_mb"] for r in all_reports),
            },
        }
        if save_report_to:
            dump_json(final, save_report_to)
            print(f"\n[M3Runner] Report saved to {save_report_to}")
        return final


def _avg(values) -> float:
    vals = list(values)
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def main():
    parser = argparse.ArgumentParser(description="Profile MiniMax M3 (MSA) KV cache behavior")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to M3 weights, or M2 weights when using --standin")
    parser.add_argument("--prompts-file", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--capture-every", type=int, default=8)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--multi-gpu", dest="multi_gpu", action="store_true", default=None,
                        help="Aggregate NVML across all GPUs (auto-on when >1 GPU)")
    parser.add_argument("--standin", action="store_true",
                        help="Treat model-path as the M2 stand-in (code-path test)")
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts = json.load(f)

    runner = MiniMaxM3Runner(
        args.model_path, device=args.device,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit,
        multi_gpu=args.multi_gpu, is_standin=args.standin,
    )
    runner.load()
    runner.run_profiled(
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        capture_every_n=args.capture_every,
        save_report_to=args.output,
    )


if __name__ == "__main__":
    main()
