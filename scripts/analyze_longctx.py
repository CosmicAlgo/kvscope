"""Analyze long-context sweep data for linearity validation."""
import json, os
import numpy as np

data = json.load(open("results_local/longctx_sweep_pythia.json"))
results = data["results"]

tokens = [r["actual_new_tokens"] for r in results]
kv_mb = [r["kv_overhead_mb"] for r in results]

# Linear fit
coeffs = np.polyfit(tokens, kv_mb, 1)
slope = coeffs[0]  # MB/token
intercept = coeffs[1]

# R-squared
kv_pred = np.polyval(coeffs, tokens)
ss_res = sum((np.array(kv_mb) - kv_pred)**2)
ss_tot = sum((np.array(kv_mb) - np.mean(kv_mb))**2)
r_squared = 1 - ss_res / ss_tot

# Quadratic fit
coeffs2 = np.polyfit(tokens, kv_mb, 2)

print("Long-Context KV Growth: Pythia-1.4B (RTX 4060)")
print("=" * 55)
print(f"{'Tokens':>7} {'KV MB':>8} {'Predicted':>10} {'Residual':>8}")
print("-" * 55)
for t, k in zip(tokens, kv_mb):
    pred = slope * t + intercept
    res = k - pred
    print(f"{t:>7} {k:>8.1f} {pred:>10.1f} {res:>+8.1f}")

print(f"\nLinear fit: KV = {slope:.4f} * tokens + {intercept:.1f}")
print(f"  Slope = {slope:.4f} MB/token = {slope*1024:.2f} KB/token")
print(f"  Per layer = {slope*1024/24:.2f} KB/tok/layer")
print(f"  R² = {r_squared:.6f}")

print(f"\nQuadratic fit: KV = {coeffs2[0]:.6e}*t² + {coeffs2[1]:.4f}*t + {coeffs2[2]:.1f}")
print(f"  Quadratic coefficient: {coeffs2[0]:.6e} MB/tok²")

# Theoretical comparison
# 8.00 KB/tok/layer * 24 layers = 192 KB/tok = 0.1875 MB/tok
theoretical_slope = 8.00 * 24 / 1024  # MB/token
print(f"\nTheoretical slope: {theoretical_slope:.4f} MB/token ({8.00*24:.0f} KB/tok)")
print(f"Measured slope:    {slope:.4f} MB/token ({slope*1024:.1f} KB/tok)")
print(f"Ratio: {slope/theoretical_slope:.3f}")

# Extrapolation to longer contexts
print(f"\nExtrapolation (linear model):")
for ctx in [16384, 32768, 65536, 131072]:
    pred_kv = slope * ctx + intercept
    print(f"  {ctx:>6} tokens: {pred_kv:>8.0f} MB ({pred_kv/1024:.1f} GB)")
