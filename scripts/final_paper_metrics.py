#!/usr/bin/env python3
"""Final paper metrics — correct KV density from actual tensor data."""
import json, os, statistics

D = os.path.join(os.path.dirname(__file__), "..", "final_data", "results_v2")
CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs_check", "results_v2", "model_configs")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_config(name):
    path = os.path.join(CONFIGS, f"{name}_config.json")
    return load_json(path) if os.path.exists(path) else {}

# ── Extract metrics per model ────────────────────────────────
def get_metrics(key):
    data = load_json(os.path.join(D, f"{key}_profile.json"))
    pp = data.get("per_prompt", data.get("results", []))
    tokens = [p.get("actual_new_tokens", 0) for p in pp]
    valid = sum(1 for t in tokens if t > 10)
    max_tokens = max(tokens) if tokens else 0
    
    # Method 1: leak_detection cache_density (mha, gemma4, qwen36)
    densities_ld = []
    cvs_ld = []
    n_layers_ld = None
    total_bytes_ld = None
    for p in pp:
        ld = p.get("leak_detection") or {}
        for f in ld.get("findings", []):
            ev = f.get("evidence", {})
            if "bytes_per_token_per_layer" in ev:
                densities_ld.append(ev["bytes_per_token_per_layer"] / 1024)
            if "cv" in ev:
                cvs_ld.append(ev["cv"])
            if "n_active_layers" in ev and n_layers_ld is None:
                n_layers_ld = ev["n_active_layers"]
            if "total_bytes_at_end" in ev and total_bytes_ld is None:
                total_bytes_ld = ev["total_bytes_at_end"]
    
    # Method 2: tracer kv_cache per_layer_mb (gptoss)
    densities_tr = []
    cvs_tr = []
    n_layers_tr = None
    total_mb_tr = None
    for p in pp:
        t = p.get("actual_new_tokens", 0)
        if t < 10:
            continue
        tracer = p.get("tracer", {})
        kv = tracer.get("kv_cache", {})
        per_layer = kv.get("per_layer_mb", {})
        if not per_layer:
            continue
        vals = list(per_layer.values())
        n = len(vals)
        total = sum(vals)
        if n_layers_tr is None:
            n_layers_tr = n
        density = total * 1024 / t / n if t > 0 and n > 0 else 0  # KB/tok/layer
        if density > 0:
            densities_tr.append(density)
            total_mb_tr = total
            mean_v = statistics.mean(vals)
            std_v = statistics.stdev(vals) if len(vals) > 1 else 0
            cvs_tr.append(std_v / mean_v if mean_v > 0 else 0)
    
    # Choose best method
    if densities_ld:
        return {
            "density": statistics.median(densities_ld),
            "cv": statistics.median(cvs_ld) if cvs_ld else 0,
            "n_layers": n_layers_ld,
            "total_kv_mb": total_bytes_ld / (1024*1024) if total_bytes_ld else 0,
            "valid": valid,
            "max_tokens": max_tokens,
            "source": "leak_detection",
        }
    elif densities_tr:
        return {
            "density": statistics.median(densities_tr),
            "cv": statistics.median(cvs_tr) if cvs_tr else 0,
            "n_layers": n_layers_tr,
            "total_kv_mb": total_mb_tr if total_mb_tr else 0,
            "valid": valid,
            "max_tokens": max_tokens,
            "source": "tracer",
        }
    else:
        # Nemotron or spec_decode — no per-layer data
        overheads = [p.get("memory", {}).get("overhead_mb", 0) for p in pp]
        valid_oh = [o for t, o in zip(tokens, overheads) if t > 10 and o > 0]
        return {
            "density": 0,
            "cv": 0,
            "n_layers": None,
            "total_kv_mb": statistics.median(valid_oh) if valid_oh else 0,
            "valid": valid,
            "max_tokens": max_tokens,
            "source": "nvml_only",
        }

# ── Compute all ───────────────────────────────────────────────
models = [
    ("mha_baseline", "Pythia-1.4B", "Dense MHA", "pythia", "A"),
    ("gemma4", "Gemma 4 (27B)", "Hybrid Local/Global", "gemma4_new", "A"),
    ("qwen36", "Qwen 3.6 (35B)", "Hybrid DeltaNet+GQA", "qwen36", "A"),
    ("gptoss", "gpt-oss (Hybrid)", "Hybrid SSM+Attention", "gptoss", "A"),
    ("nemotron", "Nemotron-H (SSM)", "Pure SSM (no KV)", "nemotron", "D"),
    ("spec_decode", "Spec Decode (70B+1B)", "Speculative Decoding", "llama31-70b", "B"),
]

print(f"{'Model':<22} {'Arch':<22} {'Valid':>5} {'Lays':>4} {'KB/tok/lay':>11} {'CV':>6} {'KV MB':>8} {'Tok':>5} {'Gr':>2}")
print("-" * 90)

all_results = []
for key, name, arch, config_name, grade in models:
    m = get_metrics(key)
    config = load_config(config_name)
    
    density_str = f"{m['density']:.2f}" if m['density'] > 0 else "--"
    cv_str = f"{m['cv']:.3f}" if m['cv'] > 0 else ("0.000" if m['source'] != 'nvml_only' else "--")
    kv_str = f"{m['total_kv_mb']:.1f}" if m['total_kv_mb'] > 0 and m['source'] != 'nvml_only' else "--"
    layers_str = str(m['n_layers']) if m['n_layers'] else "--"
    
    print(f"{name:<22} {arch:<22} {m['valid']:>3}/20 {layers_str:>4} {density_str:>11} {cv_str:>6} {kv_str:>8} {m['max_tokens']:>5} {grade:>2}")
    
    all_results.append({
        "key": key, "name": name, "arch": arch, "grade": grade,
        **m, "config_name": config_name,
    })

# ── Save final metrics ───────────────────────────────────────
env_data = {
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
}

out = {"environment": env_data, "models": all_results}
with open(os.path.join(OUT, "v2_final_metrics.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)

print(f"\nSaved to {os.path.join(OUT, 'v2_final_metrics.json')}")

# ── LaTeX table rows ──────────────────────────────────────────
print(f"\n% LaTeX table rows for paper")
for r in all_results:
    d = f"{r['density']:.2f}" if r['density'] > 0 else "--"
    cv = f"{r['cv']:.3f}" if r['source'] != 'nvml_only' else "--"
    kv = f"{r['total_kv_mb']:.1f}" if r['source'] != 'nvml_only' else "--"
    lay = str(r['n_layers']) if r['n_layers'] else "--"
    print(f"  {r['name']:<22} & {r['max_tokens']:>4} & {r['valid']:>2}/20 & {lay:>3} & {d:>6} & {cv:>6} & {kv:>8} & {r['grade']} \\\\")
