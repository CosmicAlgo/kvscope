#!/usr/bin/env python3
"""Extract paper-ready metrics from V2 final data + model configs."""
import json, os, statistics

DATA = os.path.join(os.path.dirname(__file__), "..", "final_data", "results_v2")
CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs_check", "results_v2", "model_configs")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_config(name):
    path = os.path.join(CONFIGS, f"{name}_config.json")
    if not os.path.exists(path):
        return {}
    return load_json(path)

# ── Load all profiles ─────────────────────────────────────────
profiles = {}
for fname in os.listdir(DATA):
    if fname.endswith("_profile.json"):
        key = fname.replace("_profile.json", "")
        profiles[key] = load_json(os.path.join(DATA, fname))

# ── Model configs ─────────────────────────────────────────────
config_map = {
    "gemma4": "gemma4_new",
    "gptoss": "gptoss",
    "mha_baseline": "pythia",
    "qwen36": "qwen36",
    "nemotron": "nemotron",
    "spec_decode": "llama31-70b",  # verifier config
}

def get_arch_details(config):
    """Extract arch details, handling nested configs."""
    # For MoE/multimodal, look inside text_config or language_model
    for key in ["text_config", "language_config", "language_model", "model"]:
        if key in config:
            inner = config[key]
            if isinstance(inner, dict):
                # Recurse one level
                for k2 in ["text_config", "config"]:
                    if k2 in inner and isinstance(inner[k2], dict):
                        inner = inner[k2]
                        break
                return {
                    "num_hidden_layers": inner.get("num_hidden_layers", "?"),
                    "num_attention_heads": inner.get("num_attention_heads", "?"),
                    "num_key_value_heads": inner.get("num_key_value_heads", "?"),
                    "hidden_size": inner.get("hidden_size", "?"),
                    "head_dim": inner.get("head_dim", inner.get("hidden_size", 1) // max(inner.get("num_attention_heads", 1), 1)),
                    "architectures": config.get("architectures", ["?"]),
                    "model_type": config.get("model_type", inner.get("model_type", "?")),
                }
    
    return {
        "num_hidden_layers": config.get("num_hidden_layers", "?"),
        "num_attention_heads": config.get("num_attention_heads", "?"),
        "num_key_value_heads": config.get("num_key_value_heads", "?"),
        "hidden_size": config.get("hidden_size", "?"),
        "head_dim": config.get("head_dim", config.get("hidden_size", 1) // max(config.get("num_attention_heads", 1), 1)),
        "architectures": config.get("architectures", ["?"]),
        "model_type": config.get("model_type", "?"),
    }

# ── Compute metrics per model ─────────────────────────────────
results = []

model_info = [
    ("mha_baseline", "Pythia-1.4B", "Dense MHA", "pythia"),
    ("gemma4", "Gemma 4 (27B)", "Hybrid Local/Global", "gemma4_new"),
    ("qwen36", "Qwen 3.6 (35B)", "Hybrid DeltaNet+GQA", "qwen36"),
    ("gptoss", "gpt-oss (Hybrid)", "Hybrid SSM+Attention", "gptoss"),
    ("nemotron", "Nemotron-H (SSM)", "Pure SSM (no KV)", "nemotron"),
    ("spec_decode", "Spec Decode (70B+1B)", "Speculative Decoding", "llama31-70b"),
]

for key, name, arch, config_name in model_info:
    data = profiles.get(key)
    if not data:
        continue
    
    config = load_config(config_name)
    arch_d = get_arch_details(config)
    
    pp = data.get("per_prompt", data.get("results", []))
    tokens = [p.get("actual_new_tokens", 0) for p in pp]
    tps_list = [p.get("tokens_per_sec", 0) for p in pp]
    
    # KV overhead
    overheads = []
    for p in pp:
        mem = p.get("memory", {})
        o = mem.get("overhead_mb", 0)
        overheads.append(o)
    
    valid_tok = [t for t in tokens if t > 10]
    valid_oh = [o for t, o in zip(tokens, overheads) if t > 10 and o > 0]
    valid_tps = [t for t in tps_list if t > 0]
    
    # KV density (MB/token)
    densities = []
    for t, o in zip(tokens, overheads):
        if t > 10 and o > 0:
            densities.append(o / t)
    
    # KV density per layer (KB/token/layer)
    n_layers = arch_d.get("num_hidden_layers", 0)
    if isinstance(n_layers, int) and n_layers > 0 and densities:
        density_per_layer = [d * 1024 / n_layers for d in densities]  # KB/token/layer
        med_density_per_layer = statistics.median(density_per_layer)
    else:
        med_density_per_layer = None
    
    # Post-EOS retention
    post_eos_scores = []
    for p in pp:
        leak = p.get("leak_detection", {})
        score = leak.get("post_eos_score", leak.get("overall_score", 0))
        post_eos_scores.append(score)
    
    r = {
        "key": key,
        "name": name,
        "arch": arch,
        "config": arch_d,
        "n_prompts": len(pp),
        "n_valid": len(valid_tok),
        "median_tokens": statistics.median(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
        "median_tps": statistics.median(valid_tps) if valid_tps else 0,
        "median_overhead_mb": statistics.median(valid_oh) if valid_oh else 0,
        "median_density_mb_per_tok": statistics.median(densities) if densities else 0,
        "density_kb_per_tok_per_layer": med_density_per_layer,
        "n_layers": n_layers if isinstance(n_layers, int) else "?",
        "kv_heads": arch_d.get("num_key_value_heads", "?"),
        "attn_heads": arch_d.get("num_attention_heads", "?"),
        "hidden_size": arch_d.get("hidden_size", "?"),
        "head_dim": arch_d.get("head_dim", "?"),
        "max_post_eos": max(post_eos_scores) if post_eos_scores else 0,
    }
    results.append(r)

# ── Print summary ─────────────────────────────────────────────
print("=" * 90)
print("  PAPER-READY METRICS FROM V2 DATA")
print("=" * 90)

for r in results:
    print(f"\n  {r['name']} ({r['arch']})")
    print(f"    Layers: {r['n_layers']}, KV heads: {r['kv_heads']}, "
          f"Attn heads: {r['attn_heads']}, Hidden: {r['hidden_size']}, Head dim: {r['head_dim']}")
    print(f"    Valid prompts: {r['n_valid']}/{r['n_prompts']}")
    print(f"    Median tokens: {r['median_tokens']:.0f}, Max: {r['max_tokens']}")
    print(f"    Median throughput: {r['median_tps']:.1f} tok/s")
    print(f"    Median KV overhead: {r['median_overhead_mb']:.1f} MB")
    print(f"    KV density: {r['median_density_mb_per_tok']:.3f} MB/token "
          f"({r['median_density_mb_per_tok']*1024:.1f} KB/token)")
    if r['density_kb_per_tok_per_layer'] is not None:
        print(f"    KV density per layer: {r['density_kb_per_tok_per_layer']:.2f} KB/tok/layer")
    print(f"    Max post-EOS score: {r['max_post_eos']:.3f}")

# ── Generate LaTeX table ──────────────────────────────────────
print(f"\n\n{'='*90}")
print(f"  LaTeX TABLE: Main Metrics")
print(f"{'='*90}")

# Map for table
grade_map = {
    "mha_baseline": "A",
    "gemma4": "A",
    "qwen36": "A",
    "gptoss": "A",
    "nemotron": "D",
    "spec_decode": "B",
}

for r in results:
    key = r["key"]
    dpl = f"{r['density_kb_per_tok_per_layer']:.2f}" if r['density_kb_per_tok_per_layer'] else "--"
    kv_mb = f"{r['median_overhead_mb']:.1f}" if r['median_overhead_mb'] > 0 else "--"
    leak = f"{r['max_post_eos']:.3f}"
    grade = grade_map.get(key, "?")
    tok_budget = int(r['max_tokens']) if r['max_tokens'] > 0 else "?"
    n_layers = r['n_layers']
    
    print(f"  {r['name']:<25s} & {tok_budget:>4} tok & {r['n_valid']:>2}/20 & "
          f"{n_layers:>3} & {dpl:>6} & {kv_mb:>8} & {leak:>6} & {grade} \\\\")

# ── Save metrics JSON ─────────────────────────────────────────
out_data = {
    "environment": {
        "gpu": "NVIDIA H200",
        "vram_mib": 143771,
        "driver": "590.48.01",
        "cuda": "13.1",
        "architecture": "Hopper",
        "cpu": "Intel Xeon Platinum 8592+",
        "os": "Ubuntu 22.04.5 LTS",
        "python": "3.10.12",
        "torch": "2.12.0+cu130",
        "triton": "3.7.0",
        "transformers": "5.9.0",
        "bitsandbytes": "0.49.2",
        "accelerate": "1.13.0",
    },
    "models": results,
}

out_path = os.path.join(OUT, "v2_final_metrics.json")
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2, default=str)
print(f"\n  Saved: {out_path}")
