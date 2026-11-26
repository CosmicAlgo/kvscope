# vLLM comparison harness

Profiles **MiniMax M3 (MSA)**, or the MiniMax-M2 stand-in, through a vLLM
server, complementing the in-process `KVCacheTracer`.

**Status:** implemented and unit-tested (Prometheus parsing and summaries).
Not yet run on a GPU; no results from this harness are in the repository.

## Why vLLM

- vLLM (PagedAttention) is a widely used serving engine, and it can reach long
  contexts that eager `transformers.generate` cannot.
- vLLM exposes a Prometheus `/metrics` endpoint. `vllm:kv_cache_usage_perc`
  (`vllm:gpu_cache_usage_perc` in older versions; the scraper reads either) is
  the closest **server-side KV-cache signal** — the fraction of KV blocks in use.
- Pairing that gauge with black-box latency curves gives an MSA characterisation
  without needing in-process tensors.

## What it measures

| Artifact | Source | Signal |
|---|---|---|
| `*_latency.json` | `kvscope api latency` | TTFT vs input length (prefill), ITL vs context (decode), scaling exponents |
| `*_needle.json` | `kvscope api needle` | effective-context recall across depth × length |
| `*_kv_timeseries.json` | `scrape_vllm_metrics.py` | KV-cache usage, running/waiting requests, and per-interval TTFT/TPOT averages, token rates and prefix-cache hit rate |

## Run it

```bash
# With the M2 stand-in:
MODEL_PATH=MiniMaxAI/MiniMax-M2 TP=4 MAX_MODEL_LEN=131072 \
  bash experiments/vllm_comparison/serve_and_profile.sh

# With real M3 weights (not public as of May 2026, when this was written):
MODEL_PATH=/models/minimax_m3 MODEL_NAME=MiniMax-M3 TP=4 MAX_MODEL_LEN=1000000 \
  bash experiments/vllm_comparison/serve_and_profile.sh
```

Requires `pip install vllm` (or `pip install -e '.[vllm]'`) and an NVIDIA GPU
node — see `docs/ibm_cloud_gpu_setup.md`.

## Cross-validation (planned)

With M3 weights, run **both** paths on the same prompts:

- in-process: `python -m src.models.minimax_m3_runner ...` → per-layer KV bytes,
  head utilisation, leak detectors;
- served: this harness → KV-cache usage % + latency scaling.

Comparing the two would show whether the black-box latency signature agrees
with what the in-process tracer measures.
