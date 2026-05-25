#!/usr/bin/env python3
"""Extract real KV metrics from V2 data — handles both data formats."""
import json, os, statistics

D = os.path.join(os.path.dirname(__file__), "..", "final_data", "results_v2")

def load(name):
    return json.load(open(os.path.join(D, name), encoding="utf-8"))

def extract_from_leak_detection(pp):
    """Extract from leak_detection.findings (mha_baseline, gemma4, qwen36 format)."""
    densities = []
    cvs = []
    n_layers = None
    total_bytes = None
    for p in pp:
        ld = p.get("leak_detection") or {}
        for f in ld.get("findings", []):
            ev = f.get("evidence", {})
            if "bytes_per_token_per_layer" in ev:
                densities.append(ev["bytes_per_token_per_layer"] / 1024)  # KB/tok/layer
            if "cv" in ev:
                cvs.append(ev["cv"])
            if "n_active_layers" in ev and n_layers is None:
                n_layers = ev["n_active_layers"]
            if "total_bytes_at_end" in ev and total_bytes is None:
                total_bytes = ev["total_bytes_at_end"]
    return densities, cvs, n_layers, total_bytes

def extract_from_tracer(pp):
    """Extract from tracer.kv_cache (gptoss, nemotron, spec_decode format)."""
    densities = []
    cvs = []
    n_layers = None
    total_kv_bytes = None
    
    for p in pp:
        tracer = p.get("tracer", {})
        kv = tracer.get("kv_cache", {})
        
        if not kv:
            continue
        
        # Get per-layer data
        per_layer = kv.get("per_layer", [])
        if not per_layer:
            continue
        
        if n_layers is None:
            n_layers = len(per_layer)
        
        # Compute density from per-layer bytes
        layer_bytes = []
        for layer in per_layer:
            lb = layer.get("total_bytes", layer.get("kv_bytes", 0))
            seq_len = layer.get("seq_len", layer.get("k_seq_len", 0))
            if seq_len > 0 and lb > 0:
                layer_bytes.append(lb)
        
        if layer_bytes and n_layers:
            total = sum(layer_bytes)
            # Get max seq_len
            max_seq = max(l.get("seq_len", l.get("k_seq_len", 0)) for l in per_layer)
            if max_seq > 0:
                density = total / max_seq / n_layers / 1024  # KB/tok/layer
                densities.append(density)
                total_kv_bytes = total
                
                # CV
                mean_b = statistics.mean(layer_bytes)
                if mean_b > 0:
                    std_b = statistics.stdev(layer_bytes) if len(layer_bytes) > 1 else 0
                    cvs.append(std_b / mean_b)
    
    return densities, cvs, n_layers, total_kv_bytes

# ── Process all models ────────────────────────────────────────
models = [
    ("mha_baseline", "Pythia-1.4B", "Dense MHA"),
    ("gemma4", "Gemma 4 (27B)", "Hybrid Local/Global"),
    ("qwen36", "Qwen 3.6 (35B)", "Hybrid DeltaNet+GQA"),
    ("gptoss", "gpt-oss (Hybrid)", "Hybrid SSM+Attention"),
    ("nemotron", "Nemotron-H (SSM)", "Pure SSM (no KV)"),
    ("spec_decode", "Spec Decode (70B+1B)", "Speculative Decoding"),
]

print(f"{'Model':<25} {'Arch':<25} {'Valid':>5} {'Layers':>6} {'KB/tok/lay':>11} {'CV':>6} {'KV MB':>8}")
print("-" * 90)

for key, name, arch in models:
    data = load(f"{key}_profile.json")
    pp = data.get("per_prompt", data.get("results", []))
    tokens = [p.get("actual_new_tokens", 0) for p in pp]
    valid = sum(1 for t in tokens if t > 10)
    
    # Try leak_detection format first, then tracer format
    densities, cvs, n_layers, total_bytes = extract_from_leak_detection(pp)
    if not densities:
        densities, cvs, n_layers, total_bytes = extract_from_tracer(pp)
    
    med_density = statistics.median(densities) if densities else 0
    med_cv = statistics.median(cvs) if cvs else 0
    
    # KV MB from total_bytes
    kv_mb = total_bytes / (1024 * 1024) if total_bytes else 0
    
    # Also get NVML overhead for reference
    overheads = [p.get("memory", {}).get("overhead_mb", 0) for p in pp]
    valid_oh = [o for t, o in zip(tokens, overheads) if t > 10 and o > 0]
    med_overhead = statistics.median(valid_oh) if valid_oh else 0
    
    density_str = f"{med_density:.2f}" if med_density > 0 else "--"
    cv_str = f"{med_cv:.3f}" if cvs else "--"
    kv_str = f"{kv_mb:.1f}" if kv_mb > 0 else f"({med_overhead:.1f} NVML)"
    
    print(f"{name:<25} {arch:<25} {valid:>3}/20 {str(n_layers):>6} {density_str:>11} {cv_str:>6} {kv_str:>8}")
    
    # Extra detail
    if key == "nemotron":
        print(f"  ^ Nemotron-H: NVML overhead includes model weights (SSM, no KV cache)")
    if key == "spec_decode":
        print(f"  ^ Spec decode: 12/20 prompts hit EOS immediately (base model)")
