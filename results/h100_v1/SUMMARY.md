# KVScope Cross-Architecture Run — 2026-04-27 08:47 UTC

Auto-generated paper-ready summary of one full data-collection run across the profiled model families.

## Run metadata

- **Date (UTC)**: 2026-04-27 08:47 UTC
- **Host**: `ml-ai-ubuntu-gpu-h100x1-80gb-ams3`
- **GPU**: NVIDIA H100 80GB HBM3, 81559 MiB, 590.48.01
- **Git commit**: `264d359`
- **Results directory**: `results/full_run_20260427T050945Z`
- **Models profiled**: mha_baseline, gemma4, glm51, gptoss
- **Prompts per model**: mha_baseline=15, gemma4=15, glm51=15, gptoss=15

## Cross-architecture summary table

Numbers averaged across prompts. CV (coefficient of variation) of per-layer KV bytes captures architectural heterogeneity: ~0 for uniform DynamicCache; >0.3 for hybrid attention; >0.6 for strongly bimodal (sliding+dense) caches.

| Model | KB/tok/layer | Layer CV | Head util % | Dead heads | Peak overhead (MB) | Tok/s | WikiText-103 PPL | Max leak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia-1.4B (pure MHA baseline / architectural anchor) | 8.00 | 0.000 | 100.0 | 0.0 | 98 | 87.1 | — | 0.335 |
| Gemma 4 (GQA + local/global interleaving) | 14.67 | 0.203 | 100.0 | 0.0 | 8610 | 5055.4 | 827.71 | 0.212 |
| GLM-4.7-Flash (MoE + DeepSeek Sparse Attention; proxy for GLM-5.1) | — | — | 100.0 | 0.0 | 417 | 15.1 | 15.29 | 0.329 |
| OpenAI gpt-oss-120b (sliding/full hybrid + MoE) | 2.00 | 0.935 | 100.0 | 0.0 | 57 | 14.9 | 317.66 | 0.241 |

_KB/tok/layer is the average bytes of K+V stored per generated token per attention layer at end-of-generation. Layer CV is the coefficient of variation of per-layer cache size — high CV indicates hybrid attention. Head util %, dead heads from `triton_ops.measure_kv_head_utilization`. WikiText-103 PPL from `experiments/perplexity_eval.py` (sliding-window protocol)._


## Per-model deep dive

### Pythia-1.4B (pure MHA baseline / architectural anchor)

**Representative detector output (last prompt):**

- `growth_curve` (severity=low, score=0.000): Sliding-window plateau detected — KV seq_len caps at 1171 tokens (p95=1123, p99=1163). Growth curve is saturating by design, not super-linear. Skipping linearity check.
- `fragmentation` (severity=low, score=0.000): Avg fragmentation ratio: 6.21%. Max fragmented: 293.6 MB. Fragmentation trend: stable.
- `layer_anomaly` (severity=low, score=0.000): 0 layers are statistical outliers (>3σ from mean). Mean layer KV: 9.15MB ± 0.00MB.
- `post_eos` (severity=low, score=0.000): Post-EOS unreleased memory: 0.0 MB (0.0% of peak KV overhead). Expected freed: 56.6 MB, actual freed: 56.6 MB.
- `cache_density` (severity=low, score=0.000): Cache density: 8.00 KB/token/layer (avg over 24 layers; total 192.0 KB/token; seq_len=1171 from k_seq_len).
- `layer_uniformity` (severity=low, score=0.000): Per-layer KV CV=0.000 — uniform (plain DynamicCache pattern). mean=9368.0 KB, std=0.0 KB across 24 layers (last snapshot per layer). | by type: mha: 9368.0 KB (n=24)

**Per-prompt summary:**

| # | Tokens/s | KV overhead (MB) | Leak score |
|---:|---:|---:|---:|
| 1 | 26.3 | 650.1 | 0.335 |
| 2 | 91.0 | 58.7 | 0.000 |
| 3 | 88.8 | 60.8 | 0.013 |
| 4 | 93.0 | 60.8 | 0.013 |
| 5 | 93.8 | 58.7 | 0.000 |
| 6 | 93.1 | 56.6 | 0.000 |
| 7 | 85.2 | 58.7 | 0.013 |
| 8 | 93.0 | 58.7 | 0.000 |
| 9 | 94.4 | 58.7 | 0.000 |
| 10 | 84.0 | 60.8 | 0.025 |
| 11 | 93.7 | 56.6 | 0.000 |
| 12 | 92.8 | 58.7 | 0.013 |
| 13 | 93.1 | 56.6 | 0.000 |
| 14 | 91.3 | 56.6 | 0.000 |
| 15 | 92.4 | 56.6 | 0.000 |

### Gemma 4 (GQA + local/global interleaving)

**Representative detector output (last prompt):**

- `growth_curve` (severity=low, score=0.000): Sliding-window plateau detected — KV seq_len caps at 4239 tokens (p95=4034, p99=4199). Growth curve is saturating by design, not super-linear. Skipping linearity check.
- `fragmentation` (severity=low, score=0.000): Avg fragmentation ratio: 2.94%. Max fragmented: 6493.8 MB. Fragmentation trend: growing.
- `layer_anomaly` (severity=low, score=0.000): 0 layers are statistical outliers (>3σ from mean). Mean layer KV: 60.71MB ± 12.45MB.
- `post_eos` (severity=critical, score=0.485): Post-EOS unreleased memory: 5047.8 MB (48.5% of peak KV overhead). Expected freed: 10414.5 MB, actual freed: 5366.6 MB.
- `cache_density` (severity=low, score=0.000): Cache density: 14.67 KB/token/layer (avg over 60 layers; total 880.0 KB/token; seq_len=4239 from k_seq_len).
- `layer_uniformity` (severity=low, score=0.000): Per-layer KV CV=0.203 — mildly heterogeneous (typical GQA / MoE). mean=62172.0 KB, std=12638.3 KB across 60 layers (last snapshot per layer). | by type: local: 62172.0 KB (n=60)

**Per-prompt summary:**

| # | Tokens/s | KV overhead (MB) | Leak score |
|---:|---:|---:|---:|
| 1 | 14.1 | 10416.5 | 0.180 |
| 2 | 14.0 | 9154.1 | 0.203 |
| 3 | 14.0 | 9107.9 | 0.199 |
| 4 | 13.9 | 8980.0 | 0.199 |
| 5 | 13.9 | 8684.3 | 0.205 |
| 6 | 14.0 | 8571.1 | 0.211 |
| 7 | 14.1 | 8707.4 | 0.206 |
| 8 | 14.0 | 8369.7 | 0.211 |
| 9 | 14.1 | 8313.1 | 0.212 |
| 10 | 14.2 | 8361.4 | 0.210 |
| 11 | 39901.8 | 4787.8 | 0.022 |
| 12 | 13.9 | 10699.7 | 0.179 |
| 13 | 12.9 | 9535.8 | 0.195 |
| 14 | 35748.3 | 5049.9 | 0.022 |
| 15 | 13.7 | 10414.5 | 0.179 |

### GLM-4.7-Flash (MoE + DeepSeek Sparse Attention; proxy for GLM-5.1)

**Per-prompt summary:**

| # | Tokens/s | KV overhead (MB) | Leak score |
|---:|---:|---:|---:|
| 1 | 10.7 | 5484.1 | 0.289 |
| 2 | 15.0 | 104.9 | 0.007 |
| 3 | 15.0 | 2.1 | 0.000 |
| 4 | 15.3 | 52.4 | 0.000 |
| 5 | 15.3 | 52.4 | 0.000 |
| 6 | 15.2 | 155.2 | 0.119 |
| 7 | 15.2 | 6.3 | 0.000 |
| 8 | 15.2 | 52.4 | 0.000 |
| 9 | 17.8 | 4.2 | 0.000 |
| 10 | 15.2 | 111.2 | 0.181 |
| 11 | 15.2 | 54.5 | 0.000 |
| 12 | 15.1 | 58.7 | 0.329 |
| 13 | 15.2 | 4.2 | 0.000 |
| 14 | 15.3 | 54.5 | 0.000 |
| 15 | 15.1 | 54.5 | 0.000 |

### OpenAI gpt-oss-120b (sliding/full hybrid + MoE)

**Representative detector output (last prompt):**

- `growth_curve` (severity=low, score=0.000): KV growth linearity R²=1.000 (threshold 0.95). Quadratic fit advantage: 0.000. Residual trend: +0.0000 MB/step. Max positive residual: 0.00 MB.
- `fragmentation` (severity=low, score=0.110): Avg fragmentation ratio: 18.68%. Max fragmented: 15040.2 MB. Fragmentation trend: stable.
- `layer_anomaly` (severity=low, score=0.000): 0 layers are statistical outliers (>3σ from mean). Mean layer KV: 1.65MB ± 1.42MB.
- `post_eos` (severity=low, score=0.000): Post-EOS unreleased memory: 0.0 MB (0.0% of peak KV overhead). Expected freed: 50.3 MB, actual freed: 50.3 MB.
- `cache_density` (severity=low, score=0.000): Cache density: 2.00 KB/token/layer (avg over 36 layers; total 38.9 KB/token; seq_len=1559 from k_seq_len).
- `layer_uniformity` (severity=low, score=0.000): Per-layer KV CV=0.849 — strongly bimodal (sliding+dense or pre-allocated mix). mean=1686.0 KB, std=1432.0 KB across 36 layers (last snapshot per layer). | by type: full: 3118.0 KB (n=18), sliding: 254.0 KB (n=18)

**Per-prompt summary:**

| # | Tokens/s | KV overhead (MB) | Leak score |
|---:|---:|---:|---:|
| 1 | 13.3 | 123.7 | 0.241 |
| 2 | 13.1 | 52.4 | 0.023 |
| 3 | 13.2 | 52.4 | 0.023 |
| 4 | 13.2 | 52.4 | 0.023 |
| 5 | 13.3 | 52.4 | 0.023 |
| 6 | 13.3 | 54.5 | 0.037 |
| 7 | 13.1 | 48.2 | 0.023 |
| 8 | 13.2 | 52.4 | 0.023 |
| 9 | 13.3 | 50.3 | 0.023 |
| 10 | 13.3 | 54.5 | 0.023 |
| 11 | 13.2 | 50.3 | 0.023 |
| 12 | 13.2 | 50.3 | 0.023 |
| 13 | 13.2 | 52.4 | 0.023 |
| 14 | 13.3 | 52.4 | 0.023 |
| 15 | 38.5 | 50.3 | 0.023 |

## Raw artifacts

- `comparative_analysis.json` (2.2 KB)
- `gemma4_profile.json` (14727.0 KB)
- `glm51_profile.json` (8532.5 KB)
- `gptoss_profile.json` (6073.5 KB)
- `mha_baseline_profile.json` (1356.3 KB)
- `perplexity.json` (3.1 KB)
- `prompts.json` (9.6 KB)
- `logs/comparative.log`
- `logs/env_20260427T051317Z.txt`
- `logs/env_20260427T061745Z.txt`
- `logs/env_20260427T072818Z.txt`
- `logs/env_20260427T084408Z.txt`
- `logs/env_20260427T084414Z.txt`
- `logs/full_run.log`
- `logs/perplexity.log`
- `plots/fig_density_bar.png`
- `plots/fig_layer_type_split.png`
- `plots/fig_layer_uniformity.png`
- `plots/fig_throughput_overhead.png`
- `plots/fig_throughput_vs_seqlen.png`

---
_Generated by `experiments/generate_summary.py` on 2026-04-27 08:47 UTC._