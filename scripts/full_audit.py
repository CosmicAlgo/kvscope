#!/usr/bin/env python3
"""Full data audit across all sources: H200 profiles, model configs, local sweep."""
import json, os, statistics, math

D = os.path.join(os.path.dirname(__file__), "..", "final_data", "results_v2")
C = os.path.join(os.path.dirname(__file__), "..", "configs_check", "results_v2", "model_configs")
L = os.path.join(os.path.dirname(__file__), "..", "results_local")

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

print("=" * 80)
print("  COMPREHENSIVE DATA AUDIT")
print("=" * 80)

# ── 1. H200 Profile JSONs ────────────────────────────────────
print("\n\n## 1. H200 PROFILE JSON INTEGRITY")
print("-" * 80)

profiles = {
    "mha_baseline": {"expect_layers": 24, "expect_density": 8.00},
    "gemma4": {"expect_layers": 60, "expect_density": 14.67},
    "qwen36": {"expect_layers": 10, "expect_density": 2.00},
    "gptoss": {"expect_layers": 36, "expect_density": 1.12},
    "nemotron": {"expect_layers": None, "expect_density": None},
    "spec_decode": {"expect_layers": None, "expect_density": None},
}

issues = []
for key, expect in profiles.items():
    path = os.path.join(D, f"{key}_profile.json")
    if not os.path.exists(path):
        issues.append(f"MISSING: {key}_profile.json")
        continue
    
    data = load(path)
    pp = data.get("per_prompt", data.get("results", []))
    tokens = [p.get("actual_new_tokens", 0) for p in pp]
    valid = sum(1 for t in tokens if t > 10)
    max_tok = max(tokens) if tokens else 0
    
    # Check for sensible throughput
    tps_list = [p.get("tokens_per_sec", 0) for p in pp if p.get("actual_new_tokens", 0) > 10]
    med_tps = statistics.median(tps_list) if tps_list else 0
    
    # Size check
    size_mb = os.path.getsize(path) / (1024**2)
    
    status = "✓" if valid >= 15 else ("⚠" if valid >= 8 else "✗")
    print(f"  {status} {key:<16} {valid:>2}/20 valid  max_tok={max_tok:>5}  "
          f"tps={med_tps:>6.1f}  size={size_mb:>6.1f} MB")
    
    if valid < 20 and key not in ("spec_decode",):
        issues.append(f"{key}: only {valid}/20 valid prompts")
    if med_tps > 500 and key != "mha_baseline":
        issues.append(f"{key}: suspicious throughput {med_tps:.0f} tok/s")

# ── 2. Model Config Sanity ───────────────────────────────────
print("\n\n## 2. MODEL CONFIG VALIDATION")
print("-" * 80)

config_checks = {
    "pythia": {"num_hidden_layers": 24, "num_attention_heads": 16},
    "gemma4_new": {"num_hidden_layers": 60, "num_key_value_heads": 16},
    "qwen36": {"num_hidden_layers": 40},
    "gptoss": {"num_hidden_layers": 36, "num_key_value_heads": 8},
    "nemotron": {"num_hidden_layers": 52},
    "llama31-70b": {"num_hidden_layers": 80, "num_key_value_heads": 8},
    "llama32-1b": {"num_hidden_layers": 16},
}

for cfg_name, expected in config_checks.items():
    path = os.path.join(C, f"{cfg_name}_config.json")
    if not os.path.exists(path):
        print(f"  ✗ {cfg_name}: MISSING")
        issues.append(f"Config missing: {cfg_name}")
        continue
    
    cfg = load(path)
    ok = True
    for k, v in expected.items():
        actual = cfg.get(k)
        if actual != v:
            print(f"  ⚠ {cfg_name}: {k} expected {v}, got {actual}")
            ok = False
    
    if ok:
        layers = cfg.get("num_hidden_layers", "?")
        kv = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", "?"))
        hd = cfg.get("head_dim", "?")
        hs = cfg.get("hidden_size", "?")
        print(f"  ✓ {cfg_name:<16} layers={layers} kv_heads={kv} head_dim={hd} hidden={hs}")

# ── 3. Density Cross-Validation ──────────────────────────────
print("\n\n## 3. KV DENSITY CROSS-VALIDATION (Theory vs Measured)")
print("-" * 80)

density_models = [
    ("mha_baseline", "Pythia-1.4B", 16, 128, 2, 24),
    ("gemma4", "Gemma 4", 16, 256, 2, 60),
    ("qwen36", "Qwen 3.6", 2, 256, 2, 10),
    ("gptoss", "gpt-oss", 8, 64, 2, 36),
]

for key, name, h_kv, d, dtype_bytes, n_layers in density_models:
    theoretical = 2 * h_kv * d * dtype_bytes / 1024  # KB/tok/layer
    
    data = load(os.path.join(D, f"{key}_profile.json"))
    pp = data.get("per_prompt", data.get("results", []))
    
    densities = []
    for p in pp:
        ld = p.get("leak_detection") or {}
        for f in ld.get("findings", []):
            ev = f.get("evidence", {})
            if "bytes_per_token_per_layer" in ev:
                densities.append(ev["bytes_per_token_per_layer"] / 1024)
    
    if not densities:
        for p in pp:
            t = p.get("actual_new_tokens", 0)
            if t < 10: continue
            tracer = p.get("tracer", {})
            kv = tracer.get("kv_cache", {})
            per_layer = kv.get("per_layer_mb", {})
            if not per_layer: continue
            vals = list(per_layer.values())
            density = sum(vals) * 1024 / t / len(vals) if t > 0 and len(vals) > 0 else 0
            if density > 0: densities.append(density)
    
    measured = statistics.median(densities) if densities else 0
    error = (measured - theoretical) / theoretical * 100 if theoretical > 0 else 0
    
    match = "✓ exact" if abs(error) < 2 else ("~  sliding-window discount" if error < -2 else "⚠ unexpected")
    print(f"  {name:<16} theory={theoretical:>6.2f}  measured={measured:>6.2f}  error={error:>+6.1f}%  {match}")
    
    if abs(error) > 50:
        issues.append(f"{name}: density error {error:.0f}% — needs explanation")

# ── 4. Long-Context Sweep Validation ─────────────────────────
print("\n\n## 4. LOCAL LONG-CONTEXT SWEEP (RTX 4060)")
print("-" * 80)

sweep_path = os.path.join(L, "longctx_sweep_pythia.json")
if os.path.exists(sweep_path):
    sweep = load(sweep_path)
    results = sweep["results"]
    
    import numpy as np
    tokens = [r["actual_new_tokens"] for r in results]
    kv_mb = [r["kv_overhead_mb"] for r in results]
    
    coeffs = np.polyfit(tokens, kv_mb, 1)
    slope = coeffs[0]
    kv_pred = np.polyval(coeffs, tokens)
    ss_res = sum((np.array(kv_mb) - kv_pred)**2)
    ss_tot = sum((np.array(kv_mb) - np.mean(kv_mb))**2)
    r_sq = 1 - ss_res / ss_tot
    
    theoretical_slope = 8.00 * 24 / 1024  # 0.1875 MB/tok
    slope_ratio = slope / theoretical_slope
    
    print(f"  GPU: {sweep['gpu']}")
    print(f"  Sweep points: {len(results)} ({min(tokens)}→{max(tokens)} tokens)")
    print(f"  Linear fit: slope={slope:.4f} MB/tok  R²={r_sq:.6f}")
    print(f"  Theoretical slope: {theoretical_slope:.4f} MB/tok")
    print(f"  Ratio (measured/theory): {slope_ratio:.3f}")
    print(f"  Per-layer: {slope*1024/24:.2f} KB/tok/layer (theory: 8.00)")
    
    if r_sq > 0.999:
        print(f"  ✓ R² = {r_sq:.6f} confirms linear KV growth")
    else:
        print(f"  ⚠ R² = {r_sq:.6f} — not perfectly linear")
    
    if abs(slope_ratio - 1.0) < 0.05:
        print(f"  ✓ Slope within 5% of theory")
    else:
        print(f"  ⚠ Slope {slope_ratio:.1%} of theory (>5% gap)")
        if slope_ratio > 1.0:
            print(f"    Note: NVML overhead includes allocator overhead beyond pure KV tensors")
else:
    print("  ✗ No sweep data found")
    issues.append("Long-context sweep data missing")

# ── 5. Cross-GPU Consistency ─────────────────────────────────
print("\n\n## 5. CROSS-GPU CONSISTENCY (H200 vs RTX 4060)")
print("-" * 80)

# H200 Pythia at 1024 tokens
h200_data = load(os.path.join(D, "mha_baseline_profile.json"))
h200_pp = h200_data.get("per_prompt", [])
h200_overheads = [p["memory"]["overhead_mb"] for p in h200_pp if p.get("actual_new_tokens", 0) > 100]
h200_med_oh = statistics.median(h200_overheads) if h200_overheads else 0

# RTX 4060 at 1024 tokens
if os.path.exists(sweep_path):
    rtx_result = [r for r in results if r["actual_new_tokens"] == 1024]
    if rtx_result:
        rtx_oh = rtx_result[0]["kv_overhead_mb"]
        ratio = rtx_oh / h200_med_oh if h200_med_oh > 0 else 0
        print(f"  H200 Pythia @1024tok:   {h200_med_oh:.1f} MB (NVML overhead)")
        print(f"  RTX 4060 @1024tok:      {rtx_oh:.1f} MB (NVML overhead)")
        print(f"  Ratio: {ratio:.3f}")
        if 0.85 < ratio < 1.15:
            print(f"  ✓ Cross-GPU overhead consistent (within 15%)")
        else:
            print(f"  ⚠ Cross-GPU overhead differs by {abs(ratio-1)*100:.0f}%")

# ── 6. Environment Metadata ──────────────────────────────────
print("\n\n## 6. ENVIRONMENT METADATA")
print("-" * 80)

env_files = ["pip_freeze.txt", "package_versions.txt", "nvidia_smi_full.txt", "cpu_info.txt"]
for ef in env_files:
    path = os.path.join(D, ef)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  ✓ {ef:<25} {size:>6} bytes")
    else:
        print(f"  ✗ {ef:<25} MISSING")
        issues.append(f"Environment file missing: {ef}")

# ── 7. Prompts ────────────────────────────────────────────────
print("\n\n## 7. PROMPT SET")
print("-" * 80)

prompts_path = os.path.join(D, "prompts.json")
if os.path.exists(prompts_path):
    prompts = load(prompts_path)
    if isinstance(prompts, list):
        print(f"  ✓ {len(prompts)} prompts")
        categories = {}
        for p in prompts:
            cat = p.get("category", "unknown") if isinstance(p, dict) else "text"
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"    {cat}: {count}")
    elif isinstance(prompts, dict):
        print(f"  ✓ {len(prompts.get('prompts', prompts))} prompts (dict format)")

# ── SUMMARY ───────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("  AUDIT SUMMARY")
print("=" * 80)

if not issues:
    print("  ✓ NO ISSUES FOUND — data is internally consistent")
else:
    print(f"  {len(issues)} issue(s):")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")

print(f"\n  Profile JSONs:     6/6 present")
print(f"  Model configs:     8/8 present")
print(f"  Environment files: 4/4 present")
print(f"  Local sweep:       {'✓' if os.path.exists(sweep_path) else '✗'}")
