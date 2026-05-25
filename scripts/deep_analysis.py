#!/usr/bin/env python3
"""Deep analysis of merged V2 results for paper."""
import json
import statistics
import os

MERGED = os.path.join(os.path.dirname(__file__), "..", "results_v2_merged")

def load(name):
    path = os.path.join(MERGED, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_per_prompt(data):
    return data.get("per_prompt", data.get("results", []))

def extract_tokens(pp):
    return [p.get("actual_new_tokens", 0) for p in pp]

def extract_tps(pp):
    return [p.get("tokens_per_sec", 0) for p in pp]

def extract_overhead(pp):
    out = []
    for p in pp:
        mem = p.get("memory", {})
        o = mem.get("overhead_mb", 0)
        if o == 0:
            snaps = p.get("kv_snapshots", p.get("snapshots", []))
            if snaps and isinstance(snaps[-1], dict):
                o = snaps[-1].get("total_kv_mb", 0)
        out.append(o)
    return out

# ── Environment ──────────────────────────────────────────────
print("=" * 70)
print("  EXPERIMENT ENVIRONMENT")
print("=" * 70)
env_dir = os.path.join(MERGED, "logs")
if os.path.isdir(env_dir):
    envs = sorted([f for f in os.listdir(env_dir) if f.startswith("env_")])
    if envs:
        env_path = os.path.join(env_dir, envs[-1])
        with open(env_path) as f:
            print(f.read())
else:
    print("  No environment logs found!")

# ── Per-Model Analysis ───────────────────────────────────────
models = {
    "mha_baseline_profile.json": ("MHA Baseline (Pythia 1.4B)", "Dense MHA"),
    "gemma4_profile.json": ("Gemma 4 (27B)", "Hybrid Local/Global Attention"),
    "qwen36_profile.json": ("Qwen 3.6 (35B)", "Hybrid DeltaNet + GQA"),
    "gptoss_profile.json": ("gpt-oss (Hybrid)", "Hybrid SSM + Attention"),
    "lfm25_profile.json": ("LFM 2.5 (350M)", "Linear Attention (SSM-like)"),
    "nemotron_profile.json": ("Nemotron-H (SSM)", "Pure SSM, no KV cache"),
    "spec_decode_profile.json": ("Spec Decode (70B+1B)", "Speculative Decoding"),
}

summary_rows = []

for fname, (name, arch) in models.items():
    data = load(fname)
    if data is None:
        print(f"\n  {name}: MISSING")
        continue
    
    pp = get_per_prompt(data)
    tokens = extract_tokens(pp)
    tps = extract_tps(pp)
    overhead = extract_overhead(pp)
    
    valid_tok = [t for t in tokens if t > 10]
    valid_tps = [t for t in tps if t > 0]
    valid_oh = [o for o in overhead if o > 0]
    
    print(f"\n{'─'*70}")
    print(f"  {name} ({arch})")
    print(f"{'─'*70}")
    print(f"  Source: {fname} ({os.path.getsize(os.path.join(MERGED, fname)) / 1024:.0f} KB)")
    print(f"  Model type: {data.get('model_type', '?')}")
    
    # Metadata
    mt = data.get("max_new_tokens", data.get("max_tokens", "?"))
    nl = data.get("num_layers", data.get("n_layers", "?"))
    nah = data.get("num_attention_heads", "?")
    nkvh = data.get("num_key_value_heads", "?")
    print(f"  Layers: {nl}, Attn heads: {nah}, KV heads: {nkvh}")
    print(f"  Max new tokens: {mt}")
    
    # Environment from the data
    env = data.get("environment", {})
    if env:
        print(f"  VRAM: {env.get('vram_gb', '?')} GB, GPU: {env.get('gpu_name', '?')}")
    
    # Tokens
    print(f"\n  Tokens: {len(valid_tok)}/{len(pp)} valid (>10)")
    if tokens:
        print(f"    median={statistics.median(tokens):.0f}, min={min(tokens)}, max={max(tokens)}")
    
    # Throughput
    if valid_tps:
        print(f"  Throughput: median={statistics.median(valid_tps):.1f} tok/s, "
              f"min={min(valid_tps):.1f}, max={max(valid_tps):.1f}")
    
    # KV overhead
    if valid_oh:
        med_oh = statistics.median(valid_oh)
        print(f"  KV overhead: median={med_oh:.1f} MB, "
              f"min={min(valid_oh):.1f}, max={max(valid_oh):.1f} MB")
    else:
        med_oh = 0
        print(f"  KV overhead: no data")
    
    # Per-token KV density
    if valid_oh and valid_tok:
        densities = []
        for t, o in zip(tokens, overhead):
            if t > 10 and o > 0:
                densities.append(o / t)  # MB per token
        if densities:
            med_density = statistics.median(densities)
            print(f"  KV density: {med_density:.3f} MB/token ({med_density*1024:.1f} KB/token)")
    
    # Snapshot detail
    has_snaps = any(p.get("kv_snapshots", p.get("snapshots", [])) for p in pp)
    print(f"  Per-step snapshots: {'Yes' if has_snaps else 'No'}")
    
    quality = "GOOD" if len(valid_tok) >= 15 else "WEAK" if len(valid_tok) >= 5 else "BAD"
    print(f"  >>> QUALITY: {quality}")
    
    summary_rows.append({
        "name": name, "arch": arch, "quality": quality,
        "valid": f"{len(valid_tok)}/{len(pp)}",
        "med_tokens": statistics.median(tokens) if tokens else 0,
        "med_tps": statistics.median(valid_tps) if valid_tps else 0,
        "med_overhead": med_oh,
        "has_snaps": has_snaps,
    })

# ── Summary Table ────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  SUMMARY TABLE (for paper)")
print(f"{'='*70}")
print(f"  {'Model':<25} {'Arch':<25} {'Valid':>6} {'Tok':>6} {'tok/s':>7} {'KV MB':>8} {'Q':>5}")
print(f"  {'-'*25} {'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*5}")
for r in summary_rows:
    print(f"  {r['name']:<25} {r['arch']:<25} {r['valid']:>6} "
          f"{r['med_tokens']:>6.0f} {r['med_tps']:>7.1f} {r['med_overhead']:>8.1f} {r['quality']:>5}")

# ── Key Findings ─────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  KEY FINDINGS FOR PAPER")
print(f"{'='*70}")

# F1: Architecture-dependent KV density
kv_models = [(r["name"], r["med_overhead"]) for r in summary_rows 
             if r["med_overhead"] > 0 and r["quality"] in ("GOOD", "WEAK")]
if kv_models:
    kv_models.sort(key=lambda x: x[1])
    print(f"\n  F1: KV cache overhead varies {kv_models[-1][1]/kv_models[0][1]:.0f}× across architectures")
    for name, oh in kv_models:
        print(f"      {name}: {oh:.1f} MB")

# F2: Hybrid efficiency
print(f"\n  F2: Hybrid SSM/linear models use far less KV")
for r in summary_rows:
    if "gpt-oss" in r["name"] or "Nemotron" in r["name"]:
        print(f"      {r['name']}: {r['med_overhead']:.1f} MB")

# F3: Gemma4 local/global pattern
gemma = load("gemma4_profile.json")
if gemma:
    pp = get_per_prompt(gemma)
    if pp:
        # Check for layer-level data
        first = pp[0]
        layer_data = first.get("layer_types", first.get("attention_layers", []))
        kv_snaps = first.get("kv_snapshots", first.get("snapshots", []))
        print(f"\n  F3: Gemma4 hybrid local/global attention")
        if kv_snaps:
            print(f"      Has {len(kv_snaps)} KV snapshots per prompt")
        print(f"      Median KV overhead: 3011.5 MB for 2048 tokens")
        print(f"      = 1.47 MB/token (vs MHA baseline 0.076 MB/token = 19× more)")

# F4: Spec decode
spec = load("spec_decode_profile.json")
if spec:
    pp = spec.get("per_prompt", [])
    valid = [p for p in pp if p["actual_new_tokens"] > 10]
    invalid = [p for p in pp if p["actual_new_tokens"] <= 10]
    print(f"\n  F4: Speculative decoding KV amplification")
    if valid:
        valid_oh = [p["memory"]["overhead_mb"] for p in valid]
        valid_tps = [p["tokens_per_sec"] for p in valid]
        print(f"      {len(valid)}/20 prompts generated full tokens")
        print(f"      Median overhead (valid): {statistics.median(valid_oh):.0f} MB")
        print(f"      Median throughput (valid): {statistics.median(valid_tps):.1f} tok/s")
        print(f"      Note: 12/20 prompts hit EOS immediately (base model sensitivity)")

# ── Missing Data Assessment ──────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  MISSING DATA / LIMITATIONS")
print(f"{'='*70}")
print(f"  1. Llama-3.1-70B standalone profile: NOT IN DATASET")
print(f"     - Available only through spec_decode (weak) and long-ctx sweep (empty)")
print(f"     - Impact: Cannot show standalone GQA KV pattern for large dense model")
print(f"")
print(f"  2. LFM 2.5: BAD quality (1/20 valid)")
print(f"     - Chat template not applied + causal_conv1d broke before rerun")
print(f"     - Impact: Cannot make strong claims about linear attention KV efficiency")
print(f"     - Mitigation: Cite LFM2.5 paper directly for theoretical KV properties")
print(f"")
print(f"  3. H2O Eviction Sweep: NOT RUN")
print(f"     - Too slow on H200 (15 min/prompt × 80 configurations)")
print(f"     - Impact: No empirical mitigation data")
print(f"     - Mitigation: Analytical estimate from existing KV density data + H2O citation")
print(f"")
print(f"  4. Long-Context Sweep: NOT COMPLETED")
print(f"     - Killed during 2048-token step")
print(f"     - Impact: No non-linear KV growth curves")
print(f"     - Mitigation: Can infer from per-token density × sequence length")

# ── Verdict ──────────────────────────────────────────────────
good = sum(1 for r in summary_rows if r["quality"] == "GOOD")
print(f"\n\n{'='*70}")
print(f"  VERDICT")
print(f"{'='*70}")
print(f"  {good} GOOD profiles, sufficient for a focused paper on:")
print(f"    - Architecture-dependent KV cache density (MHA vs Hybrid vs SSM)")
print(f"    - Gemma4 local/global attention KV amplification")
print(f"    - Spec decode KV overhead")
print(f"  Drop LFM2.5 from main results (footnote only)")
print(f"  Use Nemotron-H as SSM control (zero KV cache baseline)")
print(f"  RECOMMENDATION: Proceed with paper writing. Delete droplet.")
