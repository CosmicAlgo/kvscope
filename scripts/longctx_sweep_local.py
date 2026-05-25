#!/usr/bin/env python3
"""Long-context KV cache sweep for Pythia-1.4B on local RTX 4060 (8GB VRAM).

Sweeps generation lengths: 128, 256, 512, 1024, 2048, 4096, 8192
Measures KV cache memory at each length to validate linear growth model.
"""
import json, os, time, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-1.4b-deduped"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_local")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sweep lengths
SWEEP_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
PROMPT = "Write a detailed essay about the history of computing, from the abacus to modern quantum computers. Include key milestones, inventors, and the social impact of each technological revolution."

def measure_kv_footprint(model, tokenizer, prompt, max_new_tokens):
    """Run generation and measure KV cache memory at each step."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    baseline_mem = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Register hooks on attention modules to capture KV sizes
    kv_sizes = {}  # layer_idx -> list of (step, bytes)
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # Check for past_key_values in the module
            if hasattr(module, 'past_key_value') and module.past_key_value is not None:
                pkv = module.past_key_value
                if isinstance(pkv, (list, tuple)) and len(pkv) >= 2:
                    k, v = pkv[0], pkv[1]
                    if hasattr(k, 'shape') and hasattr(v, 'shape'):
                        kv_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()
                        step = k.shape[2] if len(k.shape) >= 4 else 0
                        if layer_idx not in kv_sizes:
                            kv_sizes[layer_idx] = []
                        kv_sizes[layer_idx].append((step, kv_bytes))
        return hook
    
    # Register hooks on all attention modules
    hooks = []
    for i, layer in enumerate(model.gpt_neox.layers):
        h = layer.attention.register_forward_hook(make_hook(i))
        hooks.append(h)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
    post_mem = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    generated_tokens = output.shape[1] - input_len
    
    # Extract final KV sizes per layer
    per_layer_kv = {}
    for layer_idx, snapshots in kv_sizes.items():
        if snapshots:
            # Take the last snapshot (end of generation)
            last_step, last_bytes = snapshots[-1]
            per_layer_kv[layer_idx] = {
                "step": last_step,
                "bytes": last_bytes,
                "kb": last_bytes / 1024,
                "mb": last_bytes / (1024**2),
            }
    
    # Also measure from model's past_key_values directly
    total_kv_bytes = 0
    n_active_layers = 0
    if hasattr(model, 'past_key_values') and model.past_key_values is not None:
        for i, pkv in enumerate(model.past_key_values):
            if pkv is not None and len(pkv) >= 2:
                k, v = pkv[0], pkv[1]
                if hasattr(k, 'nelement'):
                    layer_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()
                    total_kv_bytes += layer_bytes
                    n_active_layers += 1
    
    result = {
        "max_new_tokens": max_new_tokens,
        "actual_new_tokens": generated_tokens,
        "input_tokens": input_len,
        "total_seq_len": input_len + generated_tokens,
        "baseline_mem_mb": round(baseline_mem, 1),
        "peak_mem_mb": round(peak_mem, 1),
        "post_mem_mb": round(post_mem, 1),
        "kv_overhead_mb": round(peak_mem - baseline_mem, 1),
        "total_kv_bytes": total_kv_bytes,
        "total_kv_mb": round(total_kv_bytes / (1024**2), 2) if total_kv_bytes > 0 else 0,
        "n_active_layers": n_active_layers,
        "per_layer_kv": per_layer_kv,
    }
    
    # Clean up
    del output
    torch.cuda.empty_cache()
    gc.collect()
    
    return result

def main():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()
    
    print(f"Model loaded. Baseline VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print()
    
    results = []
    for length in SWEEP_LENGTHS:
        print(f"  Sweeping max_new_tokens={length}...", end=" ", flush=True)
        try:
            t0 = time.time()
            r = measure_kv_footprint(model, tokenizer, PROMPT, length)
            elapsed = time.time() - t0
            r["elapsed_s"] = round(elapsed, 1)
            results.append(r)
            
            kv_mb = r["total_kv_mb"] if r["total_kv_mb"] > 0 else r["kv_overhead_mb"]
            print(f"tokens={r['actual_new_tokens']}, KV={kv_mb:.1f} MB, peak={r['peak_mem_mb']:.0f} MB, {elapsed:.1f}s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at {length} tokens — stopping sweep")
                torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                raise
    
    # Save results
    out_path = os.path.join(RESULTS_DIR, "longctx_sweep_pythia.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else 0,
            "prompt_preview": PROMPT[:80],
            "sweep_lengths": SWEEP_LENGTHS,
            "results": results,
        }, f, indent=2)
    
    print(f"\nSaved: {out_path}")
    
    # Print summary
    print(f"\n{'Tokens':>7} {'KV MB':>8} {'KV/tok':>8} {'Peak MB':>8} {'KV/tok/lay':>11}")
    print("-" * 50)
    for r in results:
        tok = r["actual_new_tokens"]
        kv = r["total_kv_mb"] if r["total_kv_mb"] > 0 else r["kv_overhead_mb"]
        n_layers = r["n_active_layers"] if r["n_active_layers"] > 0 else 24
        kv_per_tok = kv / tok if tok > 0 else 0
        kv_per_tok_lay = kv * 1024 / tok / n_layers if tok > 0 else 0
        print(f"{tok:>7} {kv:>8.1f} {kv_per_tok:>8.3f} {r['peak_mem_mb']:>8.0f} {kv_per_tok_lay:>8.2f} KB")

if __name__ == "__main__":
    main()
