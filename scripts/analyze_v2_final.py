#!/usr/bin/env python3
"""Comprehensive V2 data quality analysis for paper readiness."""
import json
import statistics
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2")

def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    return json.load(open(path))

def analyze_profile(data, name):
    """Analyze a standard profiling result."""
    if data is None:
        print(f"\n{'='*60}")
        print(f"  {name}: FILE MISSING")
        print(f"{'='*60}")
        return None

    pp = data.get("per_prompt", data.get("results", []))
    if not pp:
        print(f"\n{'='*60}")
        print(f"  {name}: NO PER-PROMPT DATA")
        print(f"{'='*60}")
        return None

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Basic metadata
    model_type = data.get("model_type", "unknown")
    n_layers = data.get("num_layers", data.get("n_layers", "?"))
    print(f"  Model type: {model_type}, Layers: {n_layers}")
    print(f"  Max new tokens: {data.get('max_new_tokens', data.get('max_tokens', '?'))}")

    # Token generation stats
    tokens = []
    tps_list = []
    overhead_list = []
    
    for p in pp:
        t = p.get("actual_new_tokens", 0)
        tps = p.get("tokens_per_sec", 0)
        tokens.append(t)
        tps_list.append(tps)
        
        # Memory overhead
        mem = p.get("memory", {})
        overhead = mem.get("overhead_mb", 0)
        if overhead == 0:
            # Try computing from snapshots
            snaps = p.get("kv_snapshots", p.get("snapshots", []))
            if snaps:
                last = snaps[-1] if isinstance(snaps[-1], dict) else {}
                overhead = last.get("total_kv_mb", 0)
        overhead_list.append(overhead)

    valid_tokens = [t for t in tokens if t > 10]
    valid_tps = [t for t in tps_list if t > 0]
    valid_overhead = [o for o in overhead_list if o > 0]

    print(f"\n  Token Generation:")
    print(f"    Prompts: {len(pp)}")
    print(f"    Valid (>10 tok): {len(valid_tokens)}/{len(pp)}")
    if tokens:
        print(f"    Median tokens: {statistics.median(tokens):.0f}")
        print(f"    Min/Max: {min(tokens)}/{max(tokens)}")
    
    print(f"\n  Throughput:")
    if valid_tps:
        print(f"    Median: {statistics.median(valid_tps):.1f} tok/s")
        print(f"    Min/Max: {min(valid_tps):.1f}/{max(valid_tps):.1f}")
    else:
        print(f"    No valid throughput data")

    print(f"\n  KV Memory Overhead:")
    if valid_overhead:
        print(f"    Median: {statistics.median(valid_overhead):.1f} MB")
        print(f"    Min/Max: {min(valid_overhead):.1f}/{max(valid_overhead):.1f} MB")
    else:
        print(f"    No overhead data (check snapshots)")

    # KV snapshot analysis
    has_snapshots = False
    snapshot_counts = []
    density_values = []
    for p in pp:
        snaps = p.get("kv_snapshots", p.get("snapshots", []))
        if snaps:
            has_snapshots = True
            snapshot_counts.append(len(snaps))
            # Get density from last snapshot
            last = snaps[-1] if isinstance(snaps[-1], dict) else {}
            layers = last.get("per_layer", [])
            for layer in layers:
                if isinstance(layer, dict):
                    density_values.append(layer.get("kv_bytes", 0))

    print(f"\n  KV Snapshots:")
    if has_snapshots:
        print(f"    Prompts with snapshots: {len(snapshot_counts)}/{len(pp)}")
        print(f"    Median snapshots/prompt: {statistics.median(snapshot_counts):.0f}")
    else:
        print(f"    No per-step snapshots (memory-only profiling)")

    # Leak detection
    leak_count = 0
    for p in pp:
        leak = p.get("leak_detection", {})
        if leak:
            score = leak.get("overall_score", 0)
            if score > 0.5:
                leak_count += 1
    if leak_count:
        print(f"\n  Leak Detection: {leak_count}/{len(pp)} prompts flagged (score > 0.5)")
    
    quality = "GOOD" if len(valid_tokens) >= 15 else "WEAK" if len(valid_tokens) >= 5 else "BAD"
    print(f"\n  >>> DATA QUALITY: {quality} <<<")
    
    return {
        "name": name,
        "n_prompts": len(pp),
        "n_valid": len(valid_tokens),
        "median_tokens": statistics.median(tokens) if tokens else 0,
        "median_tps": statistics.median(valid_tps) if valid_tps else 0,
        "median_overhead_mb": statistics.median(valid_overhead) if valid_overhead else 0,
        "has_snapshots": has_snapshots,
        "quality": quality,
    }


def analyze_spec_decode(data):
    """Analyze speculative decoding results."""
    if data is None:
        print(f"\n{'='*60}")
        print(f"  SPEC DECODE: FILE MISSING")
        print(f"{'='*60}")
        return None

    print(f"\n{'='*60}")
    print(f"  SPECULATIVE DECODING (Llama 70B + 1B)")
    print(f"{'='*60}")
    
    pp = data.get("per_prompt", [])
    agg = data.get("aggregate", {})
    
    tokens = [p["actual_new_tokens"] for p in pp]
    valid = [t for t in tokens if t > 10]
    
    print(f"  Verifier: {data.get('verifier_path', '?')}")
    print(f"  Draft: {data.get('draft_path', '?')}")
    print(f"  Num assistant tokens: {data.get('num_assistant_tokens', '?')}")
    print(f"  Valid prompts: {len(valid)}/{len(pp)}")
    print(f"  Median tok/s: {agg.get('median_tokens_per_sec', 0)}")
    print(f"  Median KV overhead: {agg.get('median_overhead_mb', 0)} MB")
    
    for i, p in enumerate(pp):
        print(f"    Prompt {i+1}: {p['actual_new_tokens']} tok, "
              f"{p['tokens_per_sec']} tok/s, "
              f"overhead {p['memory']['overhead_mb']:.0f} MB")
    
    quality = "GOOD" if len(valid) >= 15 else "WEAK" if len(valid) >= 5 else "BAD"
    print(f"\n  >>> DATA QUALITY: {quality} <<<")
    return {"name": "spec_decode", "quality": quality, "n_valid": len(valid)}


def main():
    print("=" * 60)
    print("  KVScope V2 — Final Data Quality Report")
    print("=" * 60)

    results = []

    # Core model profiles
    models = [
        ("mha_baseline_profile.json", "MHA BASELINE (Pythia 1.4B)"),
        ("gemma4_profile.json", "GEMMA 4 (27B, Hybrid Local/Global)"),
        ("qwen36_profile.json", "QWEN 3.6 (35B, Hybrid DeltaNet+Attn)"),
        ("gptoss_profile.json", "GPT-OSS (Hybrid SSM+Attn)"),
        ("lfm25_profile.json", "LFM 2.5 (350M, Linear Attention)"),
        ("nemotron_profile.json", "NEMOTRON-H (SSM Control, no KV)"),
    ]

    for fname, name in models:
        data = load(fname)
        r = analyze_profile(data, name)
        if r:
            results.append(r)

    # Spec decode
    spec_data = load("spec_decode_profile.json")
    spec_r = analyze_spec_decode(spec_data)
    if spec_r:
        results.append(spec_r)

    # Summary table
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY — Paper Readiness")
    print(f"{'='*60}")
    print(f"  {'Model':<35} {'Valid':>6} {'Quality':>8} {'Snapshots':>10}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*10}")
    for r in results:
        snaps = "Yes" if r.get("has_snapshots", False) else "No"
        print(f"  {r['name']:<35} {r['n_valid']:>6} {r['quality']:>8} {snaps:>10}")

    good = sum(1 for r in results if r["quality"] == "GOOD")
    weak = sum(1 for r in results if r["quality"] == "WEAK")
    bad = sum(1 for r in results if r["quality"] == "BAD")

    print(f"\n  GOOD: {good}, WEAK: {weak}, BAD: {bad}")
    
    # Paper findings check
    print(f"\n\n{'='*60}")
    print(f"  KEY FINDINGS VERIFICATION")
    print(f"{'='*60}")
    
    # F1: KV density varies across architectures
    mha = load("mha_baseline_profile.json")
    gemma = load("gemma4_profile.json")
    qwen = load("qwen36_profile.json")
    gptoss = load("gptoss_profile.json")
    
    print(f"\n  F1: KV cache density varies across architectures")
    for fname, name, d in [("mha", "MHA Baseline", mha), ("gemma", "Gemma4", gemma),
                            ("qwen", "Qwen3.6", qwen), ("gptoss", "gpt-oss", gptoss)]:
        if d:
            pp = d.get("per_prompt", d.get("results", []))
            overheads = []
            for p in pp:
                mem = p.get("memory", {})
                o = mem.get("overhead_mb", 0)
                if o == 0:
                    snaps = p.get("kv_snapshots", p.get("snapshots", []))
                    if snaps and isinstance(snaps[-1], dict):
                        o = snaps[-1].get("total_kv_mb", 0)
                overheads.append(o)
            valid_o = [o for o in overheads if o > 0]
            if valid_o:
                print(f"    {name}: median KV overhead = {statistics.median(valid_o):.1f} MB")
            else:
                print(f"    {name}: no KV overhead data (check snapshot structure)")
    
    # F2: Spec decode KV amplification
    if spec_data:
        pp = spec_data.get("per_prompt", [])
        overheads = [p["memory"]["overhead_mb"] for p in pp]
        valid_o = [o for o in overheads if o > 0]
        if valid_o:
            print(f"\n  F2: Spec decode KV overhead = {statistics.median(valid_o):.1f} MB median")
    
    # F3: Hybrid models have non-uniform KV patterns
    if gemma:
        pp = gemma.get("per_prompt", gemma.get("results", []))
        if pp:
            first = pp[0]
            snaps = first.get("kv_snapshots", first.get("snapshots", []))
            if snaps:
                last = snaps[-1] if isinstance(snaps[-1], dict) else {}
                layers = last.get("per_layer", [])
                if layers:
                    kv_bytes = [l.get("kv_bytes", 0) for l in layers if isinstance(l, dict)]
                    if kv_bytes:
                        cv = statistics.stdev(kv_bytes) / statistics.mean(kv_bytes) if statistics.mean(kv_bytes) > 0 else 0
                        print(f"\n  F3: Gemma4 per-layer KV CV = {cv:.3f} (>0 = non-uniform)")

    print(f"\n\n  VERDICT: ", end="")
    if good >= 4:
        print("DATA IS PAPER-READY. Proceed with writing.")
    elif good >= 2:
        print("DATA IS USABLE with caveats. Note limitations.")
    else:
        print("DATA NEEDS MORE RUNS before paper.")


if __name__ == "__main__":
    main()
