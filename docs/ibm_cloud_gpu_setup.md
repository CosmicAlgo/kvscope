# IBM Cloud GPU provisioning for KVScope / MiniMax M3

Checklist for running KVScope's CUDA + Triton + vLLM stack on IBM Cloud.

## TL;DR

- **Use IBM Cloud VPC `gx3` GPU instances (x86_64 + CUDA).** Fully compatible.
- **Do NOT use PowerVS / Power (ppc64le).** NVIDIA dropped CUDA driver support
  for ppc64le, and KVScope's wheels (`triton`, `bitsandbytes`, `vllm`, model
  custom CUDA kernels) are x86_64-only. It will not run there.

## 1. Pick an instance (VPC `gx3` accelerated family)

IBM Cloud VPC offers (x86_64, CUDA): V100, L4, L40S, A100 80GB, **H100**,
**H200**, B300. MiniMax M3 is a ~200–400B MoE, so:

| Goal | Instance | Notes |
|---|---|---|
| Full-precision, long context | multi-GPU **H200 141GB** | most KV headroom; best for MSA toward 1M |
| Standard profiling (8-bit) | **4× H100 80GB** | matches the runner's `--load-in-8bit` path |
| Smallest quant / smoke test | single **A100 80GB** | tight; stand-in / short context only |

Check quota and capacity in your target region before provisioning.

## 2. Base image + NVIDIA driver / CUDA

- Use an Ubuntu 22.04 stock image (or IBM's GPU-ready image if available).
- Driver/CUDA matrix that IBM validates includes H100 (driver 550 / CUDA 12.4)
  and H200 (driver 570 / CUDA 12.8). Match the CUDA your torch wheel targets.

```bash
nvidia-smi            # confirm driver + all GPUs visible
nvcc --version        # confirm CUDA toolkit (if building kernels)
```

## 3. Python environment

```bash
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv git
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# CUDA 12.x torch wheel (match the node's CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# KVScope core + serving + profiling extras:
pip install -e '.[triton,vllm,observability]'
pip install pynvml         # required by NVMLSampler
```

## 4. Hugging Face auth + weights cache

```bash
export HF_TOKEN=hf_xxx                 # gated models
export HF_HOME=/data/hf                 # large cache on a data volume
# M2 stand-in:
python -c "from huggingface_hub import snapshot_download as d; d('MiniMaxAI/MiniMax-M2', local_dir='/models/minimax_m2')"
# M3 (not public as of May 2026; check before downloading):
python -c "from huggingface_hub import snapshot_download as d; d('MiniMaxAI/MiniMax-M3', local_dir='/models/minimax_m3')"
```

Attach a large block volume (≥ 1 TB) for weights + results; M3 weights are
hundreds of GB.

## 5. Sanity checks

```bash
python -m src.cli list                 # registry loads, minimax_m3 present
python -m src.cli info minimax_m3
pytest -q                              # unit suite
nvidia-smi topo -m                     # NVLink topology for tensor-parallel
```

## 6. Run

```bash
# In-process tracer (falls back to the M2 stand-in if M3 weights are absent):
MODELS_DIR=/models bash experiments/run_profiling.sh minimax_m3

# Served (vLLM) + black-box behavioural profile:
MODEL_PATH=/models/minimax_m2 TP=4 MAX_MODEL_LEN=131072 \
  bash experiments/vllm_comparison/serve_and_profile.sh

# API-only behavioural profile (no local weights, hits the live M3 API):
export KVSCOPE_API_KEY=...   # or MINIMAX_API_KEY
python -m src.cli api --provider minimax --model MiniMax-M3 \
  latency --lengths 1000,4000,16000,64000,256000 -o results/api/m3_latency.json
```

## 7. Multi-GPU notes

- KVScope's `NVMLSampler("all")` aggregates VRAM across every visible GPU; the M3
  runner enables this automatically when `torch.cuda.device_count() > 1`.
- For tensor-parallel serving, set `TP` to the GPU count and verify NVLink with
  `nvidia-smi topo -m` (PCIe-only links throttle large-model throughput).

## Why not PowerVS?

PowerVS runs on IBM Power (`ppc64le`). Current RHEL/Ubuntu releases ship no CUDA
for ppc64le and there are no current NVIDIA ppc64le drivers, so CUDA workloads
are effectively unsupported. KVScope additionally depends on x86_64-only wheels
(`triton`, `bitsandbytes`, `vllm`) and model-specific CUDA kernels. Provision a
VPC `gx3` x86 GPU instance instead.
