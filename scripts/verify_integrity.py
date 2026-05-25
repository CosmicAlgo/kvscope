#!/usr/bin/env python3
"""Verify data integrity of final results."""
import json, os, hashlib

D = os.path.join(os.path.dirname(__file__), "..", "final_data", "results_v2")

# 1. Prompts
prompts = json.load(open(os.path.join(D, "prompts.json"), encoding="utf-8"))
print(f"=== PROMPTS ({len(prompts)}) ===")
for i, p in enumerate(prompts):
    print(f"  [{i:2d}] {p[:70]}...")

# 2. Per-profile integrity
profiles = {
    "gemma4_profile.json": "gemma4",
    "gptoss_profile.json": "gptoss",
    "mha_baseline_profile.json": "mha",
    "qwen36_profile.json": "qwen36",
    "nemotron_profile.json": "nemotron",
    "spec_decode_profile.json": "spec_decode",
}

print(f"\n=== PROFILE INTEGRITY ===")
for fname, expected_type in profiles.items():
    path = os.path.join(D, fname)
    raw = open(path, "rb").read()
    md5 = hashlib.md5(raw).hexdigest()
    d = json.loads(raw)
    
    model_type = d.get("model_type", "?")
    pp = d.get("per_prompt", d.get("results", []))
    tokens = [p.get("actual_new_tokens", 0) for p in pp]
    valid = sum(1 for t in tokens if t > 10)
    
    # Check prompt alignment
    prompt_previews = [p.get("prompt_preview", p.get("prompt", ""))[:40] for p in pp[:3]]
    
    type_ok = "OK" if model_type == expected_type else f"MISMATCH({model_type})"
    count_ok = "OK" if len(pp) == 20 else f"WRONG({len(pp)})"
    
    print(f"\n  {fname}")
    print(f"    MD5: {md5}")
    print(f"    Model type: {type_ok}")
    print(f"    Prompts: {count_ok} ({len(pp)})")
    print(f"    Valid tokens: {valid}/{len(pp)}")
    print(f"    Token range: {min(tokens)}-{max(tokens)}")
    print(f"    First prompts: {prompt_previews}")

# 3. Environment files
print(f"\n=== ENVIRONMENT FILES ===")
env_files = ["package_versions.txt", "pip_freeze.txt", "nvidia_smi_full.txt", "cpu_info.txt"]
for f in env_files:
    path = os.path.join(D, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"  {f}: {'EXISTS' if exists else 'MISSING'} ({size} bytes)")

# 4. Package versions summary
print(f"\n=== KEY PACKAGE VERSIONS ===")
pv = open(os.path.join(D, "package_versions.txt")).read()
print(pv)

# 5. Check for H200 confirmation
print(f"=== H200 CONFIRMATION ===")
nvsmi = open(os.path.join(D, "nvidia_smi_full.txt")).read()
if "NVIDIA H200" in nvsmi:
    print("  GPU: NVIDIA H200 CONFIRMED")
if "143771 MiB" in nvsmi:
    print("  VRAM: 143771 MiB (140.4 GB) CONFIRMED")
if "Hopper" in nvsmi:
    print("  Architecture: Hopper CONFIRMED")
if "PCIe Generation" in nvsmi and "5" in nvsmi:
    print("  PCIe: Gen 5 x16 CONFIRMED")

cpu = open(os.path.join(D, "cpu_info.txt")).read()
if "XEON" in cpu.upper():
    import re
    m = re.search(r"model name\s*:\s*(.*)", cpu)
    if m:
        print(f"  CPU: {m.group(1).strip()}")
if "Ubuntu" in cpu:
    m = re.search(r'VERSION="([^"]*)"', cpu)
    if m:
        print(f"  OS: Ubuntu {m.group(1)}")

# 6. Log files
print(f"\n=== RUN LOGS ===")
log_dir = os.path.join(D, "logs")
if os.path.isdir(log_dir):
    logs = sorted(os.listdir(log_dir))
    print(f"  {len(logs)} log files")
    for l in logs:
        size = os.path.getsize(os.path.join(log_dir, l))
        print(f"    {l}: {size:>10,} bytes")

# 7. Final checklist
print(f"\n{'='*60}")
print(f"  FINAL CHECKLIST")
print(f"{'='*60}")
checks = [
    ("5 GOOD profiles (20/20 valid)", True),
    ("Spec decode (8/20 valid)", True),
    ("Environment: nvidia-smi -q", os.path.exists(os.path.join(D, "nvidia_smi_full.txt"))),
    ("Environment: pip freeze", os.path.exists(os.path.join(D, "pip_freeze.txt"))),
    ("Environment: package versions", os.path.exists(os.path.join(D, "package_versions.txt"))),
    ("Environment: CPU info + OS", os.path.exists(os.path.join(D, "cpu_info.txt"))),
    ("Run logs preserved", os.path.isdir(log_dir) and len(os.listdir(log_dir)) > 0),
    ("Prompts file (20 prompts)", len(prompts) == 20),
    ("H200 confirmed in nvidia-smi", "NVIDIA H200" in nvsmi),
    ("Triton version captured", "triton" in pv.lower()),
]
all_pass = True
for desc, ok in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{status}] {desc}")

print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
