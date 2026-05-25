#!/usr/bin/env bash
# =============================================================================
# run_profiling.sh — KVScope Master Experiment Runner
# =============================================================================
# Runs the full KV cache profiling pipeline across all three model families.
# Supports checkpoint/resume for GCP Spot VM preemption survival.
#
# Usage:
#   bash experiments/run_profiling.sh [all|gemma4|glm47flash|deepseek|mitigations]
#
# Environment:
#   MODELS_DIR    — path to downloaded model weights (default: ~/models)
#   RESULTS_DIR   — where to save results (default: ./results)
#   MAX_TOKENS    — max tokens per generation (default: 300)
#   CAPTURE_EVERY — capture KV snapshot every N steps (default: 5)
# =============================================================================

set -uo pipefail

# Track failures for final summary
FAILED_MODELS=()

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
MAX_TOKENS="${MAX_TOKENS:-300}"
CAPTURE_EVERY="${CAPTURE_EVERY:-5}"
CHECKPOINT_FILE="${RESULTS_DIR}/.checkpoint"

# ─── Durable logging ──────────────────────────────────────────────────────────
# Capture every run (stdout + stderr) to a timestamped file so cleared terminals
# never destroy paper-relevant evidence. Each phase gets its own log too.
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOGS_DIR"
RUN_LOG="${LOGS_DIR}/run_${RUN_TS}.log"
ENV_SNAPSHOT="${LOGS_DIR}/env_${RUN_TS}.txt"

# Snapshot reproducibility info (git, GPU, packages) into a sibling file.
{
  echo "=== KVScope run snapshot ($RUN_TS) ==="
  echo "host: $(hostname)"
  echo "user: $(whoami)"
  echo "pwd : $(pwd)"
  echo
  echo "--- git ---"
  git rev-parse HEAD 2>/dev/null || echo "(not a git repo)"
  git status --short 2>/dev/null || true
  echo
  echo "--- nvidia-smi ---"
  nvidia-smi 2>/dev/null || echo "(nvidia-smi unavailable)"
  echo
  echo "--- python / torch / transformers ---"
  python3 -c "import sys, torch, transformers; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('transformers', transformers.__version__)" 2>/dev/null || echo "(env probe failed)"
} > "$ENV_SNAPSHOT" 2>&1

# Send everything from this point on to both the terminal AND the run log.
# `tee -a` keeps appending so re-invocations within the same second concat cleanly.
exec > >(tee -a "$RUN_LOG") 2>&1
echo "[*] Logging run to: $RUN_LOG"
echo "[*] Env snapshot   : $ENV_SNAPSHOT"

# ─── Environment Activation ──────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "[*] Using virtual environment (.venv)"
elif [[ -f "../.venv/bin/activate" ]]; then
    source "../.venv/bin/activate"
    echo "[*] Using virtual environment (.venv)"
fi

mkdir -p "$RESULTS_DIR"

# ─── Checkpoint helpers ───────────────────────────────────────────────────────
# GCP Spot VMs can be preempted at any time. These functions save/restore
# progress so we don't re-run completed experiments.

checkpoint_done() {
  echo "$1" >> "$CHECKPOINT_FILE"
  echo "[✓] Checkpointed: $1"
}

is_done() {
  [[ -f "$CHECKPOINT_FILE" ]] && grep -q "^$1$" "$CHECKPOINT_FILE" 2>/dev/null
}

# ─── VRAM Detection ──────────────────────────────────────────────────────────
VRAM_GB=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print(0)
" 2>/dev/null || echo "0")

GPU_NAME=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print('No GPU')
" 2>/dev/null || echo "unknown")

echo "============================================================"
echo " KVScope — KV Cache Memory Dynamics Profiler"
echo "============================================================"
echo " GPU       : $GPU_NAME"
echo " VRAM      : ${VRAM_GB}GB"
echo " Models dir: $MODELS_DIR"
echo " Results   : $RESULTS_DIR"
echo " Max tokens: $MAX_TOKENS"
echo "============================================================"

# ─── Prompts file ─────────────────────────────────────────────────────────────
# Prefer the curated 20-prompt corpus (15 diverse + 5 stress tests) checked
# into the repo. Falls back to a tiny 5-prompt set if that file is missing.
CANONICAL_PROMPTS="experiments/configs/prompts_advanced.json"
PROMPTS_FILE="${PROMPTS_FILE:-$RESULTS_DIR/prompts.json}"
if [[ ! -f "$PROMPTS_FILE" ]]; then
  if [[ -f "$CANONICAL_PROMPTS" ]]; then
    cp "$CANONICAL_PROMPTS" "$PROMPTS_FILE"
    n_prompts=$(python3 -c "import json; print(len(json.load(open('$PROMPTS_FILE'))))")
    echo "[*] Using curated corpus ($n_prompts prompts) from $CANONICAL_PROMPTS"
  else
    cat > "$PROMPTS_FILE" <<'EOF'
[
  "Explain the transformer attention mechanism in detail, covering self-attention, multi-head attention, and the role of queries, keys, and values. Discuss how positional encodings interact with attention.",
  "Write a detailed analysis of the economic factors that led to the 2008 financial crisis, including the role of mortgage-backed securities, credit default swaps, and the regulatory environment that enabled excessive risk-taking.",
  "Describe the process of protein folding and why it is important for drug discovery. Include discussion of AlphaFold and its impact on structural biology.",
  "Compare and contrast the philosophical traditions of existentialism and absurdism, referencing key thinkers such as Sartre, Camus, and Kierkegaard. How do their views on meaning and freedom differ?",
  "Explain how modern operating systems manage virtual memory, including page tables, TLB caches, page faults, and swap space. Discuss the tradeoffs between memory-mapped files and explicit allocation."
]
EOF
    echo "[*] Default prompts saved to $PROMPTS_FILE"
  fi
fi

# ─── Experiment Functions ─────────────────────────────────────────────────────

run_gemma4() {
  local tag="gemma4_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: Gemma 4 KV Cache Profiling"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Gemma 4 Sizes: E2B, E4B, 26B (MoE), 31B (Dense)
  # H100 (80GB) strategy: Load 31B flagship dense model.
  if [[ "$VRAM_GB" -ge 70 ]]; then
    MODEL_PATH="$MODELS_DIR/gemma4_new"   # Gemma 4 31B fp16 (~62GB)
    QUANT="none"
  elif [[ "$VRAM_GB" -ge 30 ]]; then
    MODEL_PATH="$MODELS_DIR/gemma4_new"   # 31B int8 or 26B MoE
    QUANT="8bit"
  else
    MODEL_PATH="$MODELS_DIR/gemma4_new"   # E4B small variant
    QUANT="4bit"
  fi

  python3 -m src.models.gemma4_runner \
    --model "$MODEL_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --quant "$QUANT" \
    --output "$RESULTS_DIR/gemma4_profile.json" \
    --prompts-file "$PROMPTS_FILE"

  checkpoint_done "$tag"
}

run_deepseek() {
  local tag="deepseek_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: DeepSeek MLA KV Cache Profiling"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  python3 -c "
import sys
sys.path.insert(0, '.')
from src.models.dsv4_runner import DeepSeekRunner

runner = DeepSeekRunner(
    model_path='$MODELS_DIR/deepseek',
    load_in_4bit=True,
)
runner.load()

import json
with open('$PROMPTS_FILE') as f:
    prompts = json.load(f)

runner.run_profiled(
    prompts=prompts,
    max_new_tokens=$MAX_TOKENS,
    capture_every_n=$CAPTURE_EVERY,
    save_report_to='$RESULTS_DIR/deepseek_profile.json',
)
"

  checkpoint_done "$tag"
}

run_glm47flash() {
  local tag="glm47flash_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: GLM-4.7-Flash KV Profiling"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Fail fast with a helpful message if the model isn't on disk.
  GLM_PATH="$MODELS_DIR/glm47flash"
  if [[ ! -d "$GLM_PATH" || -z "$(ls -A "$GLM_PATH" 2>/dev/null)" ]]; then
    echo "[!] GLM model not found at $GLM_PATH"
    echo "    Download with:"
    echo "      python3 -c \"from huggingface_hub import snapshot_download; \\"
    echo "        snapshot_download('zai-org/GLM-4.7-Flash', local_dir='$GLM_PATH', ignore_patterns=['*.gguf'])\""
    echo "    (~60GB; requires HF_TOKEN if gated). Skipping GLM phase for now."
    return 0
  fi

  # GLM-4.7-Flash uses the same runner architecture as Gemma4
  # but with model_type="glm47flash"
  python3 -c "
import sys, json, gc, time, torch
sys.path.insert(0, '.')
# Compat shim for trust_remote_code modeling files (must precede model load)
from src.profiler import compat  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector

model_path = '$MODELS_DIR/glm47flash'
print(f'[GLM47Flash] Loading {model_path}...')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Monkey-patch torch.cat to convert Float8_e4m3fn -> bfloat16 before concat
# (PyTorch cat_cuda not implemented for Float8, but models merge Float8 weights)
import torch
_original_cat = torch.cat
_original_stack = torch.stack
def _patched_cat(tensors, dim=0, *, out=None):
    converted = []
    for t in tensors:
        if hasattr(t, 'dtype') and str(t.dtype) == 'torch.float8_e4m3fn':
            converted.append(t.to(torch.bfloat16))
        else:
            converted.append(t)
    return _original_cat(converted, dim, out=out)
def _patched_stack(tensors, dim=0, *, out=None):
    converted = []
    for t in tensors:
        if hasattr(t, 'dtype') and str(t.dtype) == 'torch.float8_e4m3fn':
            converted.append(t.to(torch.bfloat16))
        else:
            converted.append(t)
    return _original_stack(converted, dim, out=out)
torch.cat = _patched_cat
torch.stack = _patched_stack

import os
# GLM-4.7-Flash: keep on-disk Float8 quantization (torch_dtype='auto').
# Forcing bf16 explodes to ~200GB and triggers Linux OOM killer during
# loading. Float8 weights are small; accelerate places them all on GPU.
# Per-layer decompression during forward is slower but fits in 141GB.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    torch_dtype='auto',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print('[GLM47Flash] Model loaded')

with open('$PROMPTS_FILE') as f:
    prompts = json.load(f)

nvml = NVMLSampler()
all_reports = []

for idx, prompt in enumerate(prompts):
    print(f'[GLM47Flash] Prompt {idx+1}/{len(prompts)}')
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    # GLM-4.7-Flash doesn't support token_type_ids - filter it out
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    baseline = nvml.sample()['used_mb']
    nvml.set_baseline()

    tracer = KVCacheTracer(model, model_type='glm47flash', capture_every_n_steps=$CAPTURE_EVERY, verbose=True)
    with tracer:
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=$MAX_TOKENS, do_sample=False, use_cache=True)
        gen_time = time.time() - t0

    actual_new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    peak = nvml.sample()['used_mb']
    del inputs, outputs; torch.cuda.empty_cache(); gc.collect()
    post_eos = nvml.sample()['used_mb']

    report = tracer.report()
    df = tracer.snapshots_as_dataframe()
    detector = KVLeakDetector('glm47flash')
    leak = detector.analyze(df, baseline, peak, post_eos)
    detector.print_report(leak)

    all_reports.append({
        'prompt_idx': idx,
        'actual_new_tokens': int(actual_new_tokens),
        'generation_time_s': round(gen_time, 2),
        'tokens_per_sec': round(float(actual_new_tokens) / gen_time, 1) if gen_time > 0 else 0,
        'tracer': report,
        'leak_score': leak.overall_leak_score,
        'memory': {
            'baseline_mb': round(baseline, 2),
            'peak_mb': round(peak, 2),
            'post_eos_mb': round(post_eos, 2),
            'overhead_mb': round(peak - baseline, 2),
        },
    })

from src.profiler.json_utils import dump_json
dump_json({'model_type': 'glm47flash', 'per_prompt': all_reports}, '$RESULTS_DIR/glm47flash_profile.json')
print('[GLM47Flash] Done')
"

  checkpoint_done "$tag"
}

run_gptoss() {
  local tag="gptoss_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: OpenAI gpt-oss-120b (MoE + sliding/full hybrid)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  GPTOSS_PATH="$MODELS_DIR/gptoss"
  if [[ ! -d "$GPTOSS_PATH" || -z "$(ls -A "$GPTOSS_PATH" 2>/dev/null)" ]]; then
    echo "[!] gpt-oss model not found at $GPTOSS_PATH"
    echo "    Download with:"
    echo "      python3 -c \"from huggingface_hub import snapshot_download; \\"
    echo "        snapshot_download('openai/gpt-oss-120b', local_dir='$GPTOSS_PATH', ignore_patterns=['*.gguf','original/*'])\""
    echo "    Skipping gpt-oss phase."
    return 0
  fi

  # gpt-oss-120b is HF-native (model_type='gpt_oss'). MXFP4 weights load
  # natively in transformers >= 4.45 — no BitsAndBytesConfig needed. The
  # model is ~120GB on disk but only ~63GB in VRAM at MXFP4; we still pass
  # max_memory to allow CPU offload of any extra reference shards.
  python3 -c "
import sys, json, gc, time, torch
sys.path.insert(0, '.')
# Compat shim MUST import before transformers — installs torch.accelerator
# fallback required by the MXFP4 quantizer on torch<2.5.
from src.profiler import compat  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.profiler.kv_tracer import KVCacheTracer, NVMLSampler
from src.profiler.leak_detector import KVLeakDetector
from src.profiler.json_utils import dump_json

model_path = '$GPTOSS_PATH'
print(f'[gpt-oss] Loading {model_path}...')

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Monkey-patch missing CUDA property for kernels 0.13.0 + PyTorch 2.5.1 compat
import torch
_orig_get_device_props = torch.cuda.get_device_properties
class _PropsWrapper:
    def __init__(self, p): self._p = p
    def __getattr__(self, name):
        if name == 'shared_memory_per_block_optin': return 0
        return getattr(self._p, name)
def _patched_get_device_props(device):
    props = _orig_get_device_props(device)
    if not hasattr(props, 'shared_memory_per_block_optin'):
        return _PropsWrapper(props)
    return props
torch.cuda.get_device_properties = _patched_get_device_props

import os
# Model has on-disk Mxfp4Config which conflicts with BnB. Use native MXFP4
# with offload_folder for disk spill. Let device_map='auto' handle CPU offload
# naturally to avoid KeyError with disk-only shards.
_offload_dir = os.path.join('$RESULTS_DIR', 'offload_gptoss')
os.makedirs(_offload_dir, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    torch_dtype='auto',  # honor on-disk MXFP4
    offload_folder=_offload_dir,  # disk spill safety net
)
model.eval()
print(f'[gpt-oss] Model loaded. dtype={model.dtype}')

# Print architecture summary
config = model.config
for attr in ['num_hidden_layers','num_attention_heads','num_key_value_heads',
             'hidden_size','sliding_window','num_local_experts','num_experts_per_tok']:
    val = getattr(config, attr, 'N/A')
    print(f'    {attr}: {val}')

with open('$PROMPTS_FILE') as f:
    prompts = json.load(f)

nvml = NVMLSampler()
all_reports = []

for idx, prompt in enumerate(prompts):
    print(f'[gpt-oss] Prompt {idx+1}/{len(prompts)}')
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    baseline = nvml.sample()['used_mb']
    nvml.set_baseline()

    tracer = KVCacheTracer(model, model_type='gptoss', capture_every_n_steps=$CAPTURE_EVERY, verbose=True)
    with tracer:
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=$MAX_TOKENS, do_sample=False, use_cache=True)
        gen_time = time.time() - t0

    actual_new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    peak = nvml.sample()['used_mb']
    del inputs, outputs; torch.cuda.empty_cache(); gc.collect()
    post_eos = nvml.sample()['used_mb']

    report = tracer.report()
    df = tracer.snapshots_as_dataframe()
    detector = KVLeakDetector('gptoss')
    leak = detector.analyze(df, baseline, peak, post_eos)
    detector.print_report(leak)

    all_reports.append({
        'prompt_idx': idx,
        'actual_new_tokens': int(actual_new_tokens),
        'generation_time_s': round(gen_time, 2),
        'tokens_per_sec': round(float(actual_new_tokens) / gen_time, 1) if gen_time > 0 else 0,
        'tracer': report,
        'leak_score': leak.overall_leak_score,
        'findings': [
            {'detector': f.detector, 'severity': f.severity, 'score': f.score,
             'description': f.description, 'evidence': f.evidence}
            for f in leak.findings
        ],
        'memory': {
            'baseline_mb': round(baseline, 2),
            'peak_mb': round(peak, 2),
            'post_eos_mb': round(post_eos, 2),
            'overhead_mb': round(peak - baseline, 2),
        },
    })

dump_json({'model_type': 'gptoss', 'per_prompt': all_reports}, '$RESULTS_DIR/gptoss_profile.json')
print('[gpt-oss] Done')
"

  checkpoint_done "$tag"
}

run_nemotron() {
  local tag="nemotron_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: NVIDIA Nemotron (Mamba SSM + Sparse GQA)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  NEMOTRON_PATH="${MODELS_DIR:-/root/models}/nemotron"
  if [[ ! -d "$NEMOTRON_PATH" || -z "$(ls -A "$NEMOTRON_PATH" 2>/dev/null)" ]]; then
    echo "[!] Nemotron model not found at $NEMOTRON_PATH"
    echo "    Download with:"
    echo "      python3 -c \"from huggingface_hub import snapshot_download; \\"
    echo "        snapshot_download('nvidia/Nemotron-Cascade-2-30B-A3B', local_dir='$NEMOTRON_PATH')\""
    echo "    Skipping Nemotron phase."
    return 0
  fi

  python3 -m src.models.nemotron_runner \
    --model-path "$NEMOTRON_PATH" \
    --prompts-file "$PROMPTS_FILE" \
    --max-tokens "$MAX_TOKENS" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$RESULTS_DIR/nemotron_profile.json" \
    --ssm-baseline

  checkpoint_done "$tag"
}

run_lfm25() {
  local tag="lfm25_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: LiquidAI LFM2.5-350M (LIV convolution + GQA hybrid)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  LFM25_PATH="${MODELS_DIR:-/root/models}/lfm25_new"
  if [[ ! -d "$LFM25_PATH" || -z "$(ls -A "$LFM25_PATH" 2>/dev/null)" ]]; then
    echo "[!] LFM2.5-350M model not found at $LFM25_PATH"
    echo "    Download with:"
    echo "      python3 -c \"from huggingface_hub import snapshot_download; \\"
    echo "        snapshot_download('LiquidAI/LFM2.5-350M', local_dir='$LFM25_PATH')\""
    echo "    Skipping LFM2.5 phase."
    return 0
  fi

  python3 -m src.models.lfm25_runner \
    --model-path "$LFM25_PATH" \
    --prompts-file "$PROMPTS_FILE" \
    --max-tokens "$MAX_TOKENS" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$RESULTS_DIR/lfm25_profile.json"

  checkpoint_done "$tag"
}

run_deepseek_v4() {
  local tag="deepseek_v4_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: DeepSeek V4-Flash (CSA/HCA hybrid)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Check both possible download locations
  DSV4_PATH="${MODELS_DIR:-/root/models}/deepseek_v4"
  if [[ ! -d "$DSV4_PATH" || -z "$(ls -A "$DSV4_PATH" 2>/dev/null)" ]]; then
    # Try alternative path where hf download may have placed it
    DSV4_PATH="/data/models/deepseek_v4_flash"
  fi
  if [[ ! -d "$DSV4_PATH" || -z "$(ls -A "$DSV4_PATH" 2>/dev/null)" ]]; then
    echo "[!] DeepSeek V4 model not found at $DSV4_PATH"
    echo "    Download with:"
    echo "      python3 -c \"from huggingface_hub import snapshot_download; \\"
    echo "        snapshot_download('deepseek-ai/DeepSeek-V4-Flash-Base', local_dir='$DSV4_PATH')\""
    echo "    Skipping DeepSeek V4 phase."
    return 0
  fi

  # DeepSeek V4-Flash-Base is ~284B params. In native fp8 / bf16 that does
  # not fit on a 141GB H200 in pure GPU memory, so we load in 8-bit via
  # bitsandbytes. This still lets transformers allocate KV cache on GPU while
  # keeping weights compressed enough to run.
  local extra_args=()
  if [[ "$VRAM_GB" -lt 200 ]]; then
    extra_args+=(--load-in-8bit)
    echo "[*] H200/H100 detected (<200GB VRAM) — enabling 8-bit weight loading for DeepSeek V4"
  fi

  python3 -m src.models.deepseek_v4_runner \
    --model-path "$DSV4_PATH" \
    --prompts-file "$PROMPTS_FILE" \
    --max-tokens "$MAX_TOKENS" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$RESULTS_DIR/deepseek_v4_profile.json" \
    "${extra_args[@]}"

  checkpoint_done "$tag"
}

run_mitigations() {
  local tag="mitigations_benchmark"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: KV Cache Mitigations Benchmark"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  python3 -c "
import sys, json, torch
sys.path.insert(0, '.')
from src.mitigations.mitigations import benchmark_all_mitigations

# Generate synthetic K/V for benchmarking
B, H, S, D = 1, 32, 512, 128
k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
attn = torch.softmax(torch.randn(B, H, S, S, device='cuda'), dim=-1)

results = benchmark_all_mitigations(k, v, attn)

output = []
for r in results:
    output.append({
        'strategy': r.strategy,
        'memory_reduction_pct': r.memory_reduction_pct,
        'memory_saved_mb': r.memory_saved_mb,
        'throughput_delta_pct': r.throughput_delta_pct,
        'perplexity_delta': r.perplexity_delta,
        'config': r.config,
    })

from src.profiler.json_utils import dump_json
dump_json(output, '$RESULTS_DIR/mitigations_benchmark.json')
print('[Mitigations] Benchmark saved')
"

  checkpoint_done "$tag"
}

run_comparative() {
  local tag="comparative_analysis"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Generating Comparative Analysis"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  python3 -c "
import json, os

results_dir = '$RESULTS_DIR'
models = {}

# All known model families. Files that don't exist are silently skipped, so
# this works whether you've profiled 3 or 6 of them.
# 'mha_baseline' uses a slightly different filename (matches the runner output).
MODEL_FAMILIES = ['mha_baseline', 'gemma4', 'glm47flash', 'deepseek', 'gptoss', 'nemotron', 'lfm25', 'deepseek_v4']
for name in MODEL_FAMILIES:
    path = os.path.join(results_dir, f'{name}_profile.json')
    if os.path.exists(path):
        with open(path) as f:
            models[name] = json.load(f)

if not models:
    print('[!] No results found. Run individual experiments first.')
    exit(0)

# Build comparison table
comparison = {
    'models_compared': list(models.keys()),
    'kv_architecture': {
        'mha_baseline': 'Pure multi-head attention (Pythia-1.4B) — anchor / no GQA / no MLA',
        'gemma4':       'GQA + Shared KV + Local/Global interleave (K==V in global layers)',
        'glm47flash':        'MoE + DeepSeek Sparse Attention (DSA proxy via GLM-4.7-Flash)',
        'deepseek':     'MLA (Multi-Head Latent Attention) — compressed latent KV',
        'gptoss':       'GQA + sliding/full hybrid (alternating layers) + MoE FFN',
        'nemotron':     'Nemotron-H: Mamba SSM majority + sparse GQA attention',
        'lfm25':        'LFM2.5: LIV convolution + GQA hybrid (10 LIV blocks + 6 GQA blocks)',
        'deepseek_v4':  'DeepSeek V4: CSA/HCA hybrid (4x/128x compression + 2% KV cache)',
    },
    'per_model': {},
}

def _findings_by_detector(prompt_block):
    # Two JSON layouts coexist:
    #   gemma4/glm47flash/dsv4 runners → prompt['leak_detection']['findings']
    #   inline gptoss/nemotron in this script → prompt['findings']
    # Try both.
    findings = (
        (prompt_block.get('leak_detection') or {}).get('findings')
        or prompt_block.get('findings')
        or []
    )
    return {f.get('detector'): f for f in findings if isinstance(f, dict)}

def _overall_score(prompt_block):
    return (
        (prompt_block.get('leak_detection') or {}).get('overall_score')
        if (prompt_block.get('leak_detection') or {}).get('overall_score') is not None
        else prompt_block.get('leak_score', 0)
    )

def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None

for name, data in models.items():
    prompts = data.get('per_prompt', [])
    if not prompts:
        continue

    avg_throughput = sum(p.get('tokens_per_sec', 0) for p in prompts) / len(prompts)
    avg_overhead = sum(p.get('memory', {}).get('overhead_mb', 0) for p in prompts) / len(prompts)
    max_leak = max(_overall_score(p) for p in prompts)

    # Pull cross-architecture descriptors from the per-prompt findings.
    densities, cvs = [], []
    n_active_layers, peak_seq_lens = [], []
    full_kb_means, sliding_kb_means = [], []
    for p in prompts:
        f = _findings_by_detector(p)
        cd = (f.get('cache_density') or {}).get('evidence', {}) or {}
        lu = (f.get('layer_uniformity') or {}).get('evidence', {}) or {}
        if 'bytes_per_token_per_layer' in cd:
            densities.append(cd['bytes_per_token_per_layer'] / 1024.0)  # KB
        if 'n_active_layers' in cd:
            n_active_layers.append(cd['n_active_layers'])
        if 'peak_seq_len' in cd:
            peak_seq_lens.append(cd['peak_seq_len'])
        if 'cv' in lu:
            cvs.append(lu['cv'])
        # Per-layer-type means (for hybrid models): parse from description if available.
        desc = (f.get('layer_uniformity') or {}).get('description') or ''
        if 'full:' in desc:
            try:
                full_kb_means.append(float(desc.split('full:')[1].split('KB')[0].strip()))
            except Exception:
                pass
        if 'sliding:' in desc:
            try:
                sliding_kb_means.append(float(desc.split('sliding:')[1].split('KB')[0].strip()))
            except Exception:
                pass

    # Pull head-utilization summary (Triton kernel output) from the tracer report
    util_ratios = []
    dead_heads_total = []
    n_layers_analyzed = []
    for p in prompts:
        hu = (p.get('tracer') or {}).get('head_utilization_summary') or {}
        if 'avg_head_utilization_ratio' in hu:
            util_ratios.append(hu['avg_head_utilization_ratio'])
        if 'dead_heads_across_layers' in hu:
            dead_heads_total.append(hu['dead_heads_across_layers'])
        if 'n_layers_analyzed' in hu:
            n_layers_analyzed.append(hu['n_layers_analyzed'])

    entry = {
        'avg_tokens_per_sec':       round(avg_throughput, 1),
        'avg_kv_overhead_mb':       round(avg_overhead, 1),
        'max_leak_score':           round(max_leak, 3),
        # Cross-architecture descriptors (paper table)
        'avg_density_kb_per_tok_per_layer': _avg(densities),
        'avg_layer_cv':                     _avg(cvs),
        'avg_n_active_layers':              _avg(n_active_layers),
        'avg_peak_seq_len':                 _avg(peak_seq_lens),
        # Triton head-utilization (per-head L2 dead-head detection)
        'avg_head_utilization_ratio':       _avg(util_ratios),
        'avg_dead_heads':                   _avg(dead_heads_total),
        'avg_n_layers_with_head_data':      _avg(n_layers_analyzed),
    }
    if full_kb_means:
        entry['avg_full_layer_kb'] = _avg(full_kb_means)
    if sliding_kb_means:
        entry['avg_sliding_layer_kb'] = _avg(sliding_kb_means)

    comparison['per_model'][name] = entry

# Merge perplexity numbers (if available) into per_model so the cross-arch
# table contains a single row per model with everything you'd put in a paper.
ppl_path = os.path.join(results_dir, 'perplexity.json')
if os.path.exists(ppl_path):
    try:
        with open(ppl_path) as f:
            ppl_doc = json.load(f)
        for r in ppl_doc.get('results', []):
            label = r.get('label')
            if label and label in comparison['per_model'] and 'perplexity' in r:
                comparison['per_model'][label]['wikitext103_ppl'] = r['perplexity']
                comparison['per_model'][label]['wikitext103_n_tokens'] = r.get('n_tokens_evaluated')
    except Exception as e:
        print(f'[!] Could not merge perplexity: {e}')

from src.profiler.json_utils import dump_json, dumps_json
dump_json(comparison, os.path.join(results_dir, 'comparative_analysis.json'))

print('Comparative Analysis:')
print(dumps_json(comparison))
"

  checkpoint_done "$tag"
}

# ─── MHA baseline ─────────────────────────────────────────────────────────────
# Pure multi-head-attention anchor (Pythia-1.4B-deduped). Per-token-per-layer
# KV is exactly num_attention_heads * head_dim * 2 (K+V) * dtype_size; every
# other model's lower density is the GQA / MQA / MLA win.

run_mha_baseline() {
  local tag="mha_baseline_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: MHA Baseline (Pythia-1.4B-deduped, pure MHA)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Pythia-1.4B has max_position=2048; cap MAX_TOKENS for this phase
  local mha_max_tokens=$MAX_TOKENS
  if [[ $mha_max_tokens -gt 1024 ]]; then
    mha_max_tokens=1024
    echo "[*] MAX_TOKENS=$MAX_TOKENS exceeds Pythia max_position=2048; "
    echo "    capping baseline to 1024 generated tokens."
  fi

  python3 -m src.models.mha_baseline_runner \
    --model "${MHA_MODEL_ID:-EleutherAI/pythia-1.4b-deduped}" \
    --max-tokens "$mha_max_tokens" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$RESULTS_DIR/mha_baseline_profile.json" \
    --prompts-file "$PROMPTS_FILE"

  checkpoint_done "$tag"
}

# ─── Llama baseline ───────────────────────────────────────────────────────────
# Llama-3.1-70B dense transformer (GQA for 70B+, fits in 141GB H200 at bf16)

run_llama() {
  local tag="llama_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: Llama-3.1-70B KV Cache Profiling"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local MODEL_PATH="$MODELS_DIR/llama"
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[!] Llama model not found at $MODEL_PATH; skipping."
    echo "    Download with: huggingface-cli download meta-llama/Llama-3.1-70B --local-dir $MODEL_PATH"
    return
  fi

  python3 -m src.models.llama_runner \
    --model-path "$MODEL_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$RESULTS_DIR/llama_profile.json" \
    --prompts-file "$PROMPTS_FILE"

  checkpoint_done "$tag"
}

# ─── Llama 4 baseline ───────────────────────────────────────────────────────────
# Llama-4-Scout MoE transformer (17B active out of 16 experts, fits in 141GB H200 at bf16)

run_llama4() {
  local tag="llama4_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: Llama-4-Scout KV Cache Profiling (MoE)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local MODEL_PATH="$MODELS_DIR/llama4"
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[!] Llama 4 model not found at $MODEL_PATH; skipping."
    echo "    Download with: hf download meta-llama/Llama-4-Scout-17B-16E --local-dir $MODELS_DIR/llama4"
    return
  fi

  # Llama 4 Scout is ~109B total params (217GB bf16); needs 4-bit on H200
  local extra_args=()
  if [[ "$VRAM_GB" -lt 200 ]]; then
    extra_args+=(--load-in-4bit)
    echo "[*] H200 detected (<200GB VRAM) — enabling 4-bit NF4 for Llama 4 Scout"
  fi

  python3 -m src.models.llama4_runner \
    --model-path "$MODEL_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$RESULTS_DIR/llama4_profile.json" \
    --prompts-file "$PROMPTS_FILE" \
    "${extra_args[@]}"

  checkpoint_done "$tag"
}

# ─── Qwen3 baseline ───────────────────────────────────────────────────────────
# Qwen3-32B dense transformer (standard GQA, fits in 141GB H200 at bf16)

run_qwen3() {
  local tag="qwen3_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: Qwen3-32B KV Cache Profiling"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local MODEL_PATH="$MODELS_DIR/qwen3"
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[!] Qwen3 model not found at $MODEL_PATH; skipping."
    echo "    Download with: huggingface-cli download Qwen/Qwen3-32B --local-dir $MODEL_PATH"
    return
  fi

  python3 -m src.models.qwen3_runner \
    --model-path "$MODEL_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$RESULTS_DIR/qwen3_profile.json" \
    --prompts-file "$PROMPTS_FILE"

  checkpoint_done "$tag"
}

# ─── Qwen3.6 baseline ───────────────────────────────────────────────────────────
# Qwen3.6-27B hybrid transformer (DeltaNet + Gated Attention, fits in 141GB H200 at bf16)

run_qwen36() {
  local tag="qwen36_profile"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: Qwen3.6-27B KV Cache Profiling"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Note: Qwen3.6 uses hybrid architecture:"
  echo "   - 48 Gated DeltaNet layers (no traditional KV cache)"
  echo "   - 16 Gated Attention layers (GQA with 4 KV heads)"

  local MODEL_PATH="$MODELS_DIR/qwen36"
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[!] Qwen3.6 model not found at $MODEL_PATH; skipping."
    echo "    Download with: hf download Qwen/Qwen3.6-27B --local-dir $MODEL_PATH"
    return
  fi

  python3 -m src.models.qwen36_runner \
    --model-path "$MODEL_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$RESULTS_DIR/qwen36_profile.json" \
    --prompts-file "$PROMPTS_FILE"

  checkpoint_done "$tag"
}

# ─── Perplexity evaluation ────────────────────────────────────────────────────
# Real WikiText-103 perplexity for each profiled model. Replaces the rough
# L2-error proxy that previously occupied the "Quality Impact" column.

run_perplexity() {
  local tag="perplexity_eval"
  if is_done "$tag"; then
    echo "[SKIP] $tag already complete"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " EXPERIMENT: Perplexity Evaluation (WikiText-103)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Build the --models list dynamically based on which model directories
  # actually exist. This avoids spending wall-clock time loading models that
  # aren't present on this host.
  local model_args=()
  [[ -d "$MODELS_DIR/gemma4"     ]] && model_args+=("gemma4=$MODELS_DIR/gemma4")
  [[ -d "$MODELS_DIR/glm47flash" ]] && model_args+=("glm47flash=$MODELS_DIR/glm47flash")
  [[ -d "$MODELS_DIR/gptoss"     ]] && model_args+=("gptoss=$MODELS_DIR/gptoss")
  [[ -d "$MODELS_DIR/deepseek"   ]] && model_args+=("deepseek=$MODELS_DIR/deepseek")
  [[ -d "$MODELS_DIR/deepseek-v4" ]] && model_args+=("deepseek=$MODELS_DIR/deepseek-v4")
  # MHA baseline always evaluated (small enough to download on the fly)
  model_args+=("mha=${MHA_MODEL_ID:-EleutherAI/pythia-1.4b-deduped}")

  if [[ ${#model_args[@]} -eq 0 ]]; then
    echo "[!] No models found for perplexity eval; skipping."
    return
  fi

  echo "[*] Evaluating perplexity for: ${model_args[*]}"

  # PPL_QUANT_TIERS controls which quantization variants we run. Default
  # 'none 8bit' produces the paired baseline-vs-INT8 comparison the paper
  # needs for the mitigation table. Override to 'none 8bit 4bit' for a
  # three-tier sweep, or 'none' to just get the baseline.
  read -ra _ppl_quant <<< "${PPL_QUANT_TIERS:-none 8bit}"

  python3 experiments/perplexity_eval.py \
    --models "${model_args[@]}" \
    --output "$RESULTS_DIR/perplexity.json" \
    --context-len "${PPL_CONTEXT_LEN:-1024}" \
    --stride "${PPL_STRIDE:-512}" \
    --max-chunks "${PPL_MAX_CHUNKS:-64}" \
    --quant "${_ppl_quant[@]}"

  checkpoint_done "$tag"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

PHASE="${1:-all}"

case "$PHASE" in
  gemma4)        run_gemma4 ;;
  deepseek)      run_deepseek ;;
  deepseek_v4)   run_deepseek_v4 ;;
  dsv4)          run_deepseek_v4 ;;       # alias
  glm47flash)         run_glm47flash ;;
  gptoss)        run_gptoss ;;
  nemotron)      run_nemotron ;;
  lfm25)         run_lfm25 ;;
  qwen3)         run_qwen3 ;;
  qwen36)        run_qwen36 ;;
  llama)         run_llama ;;
  llama4)        run_llama4 ;;
  mha_baseline)  run_mha_baseline ;;
  mha)           run_mha_baseline ;;     # alias
  mitigations)   run_mitigations ;;
  comparative)   run_comparative ;;
  perplexity)    run_perplexity ;;
  ppl)           run_perplexity ;;       # alias
  all)
    echo ""
    echo "[*] Running ALL experiments (with checkpoint/resume)"
    echo "[*] If the VM is preempted, restart and re-run. Completed"
    echo "    experiments will be skipped automatically."
    echo ""
    # Run each model with error trapping so one failure doesn't kill the rest
    for phase_fn in run_mha_baseline run_lfm25 run_llama4 run_qwen36 run_gemma4 run_nemotron run_gptoss; do
      if ! $phase_fn; then
        echo "[!] FAILED: $phase_fn — continuing with remaining models"
        FAILED_MODELS+=("$phase_fn")
      fi
    done
    run_comparative
    echo ""
    echo "============================================================"
    if [[ ${#FAILED_MODELS[@]} -eq 0 ]]; then
      echo " ALL EXPERIMENTS COMPLETE"
    else
      echo " EXPERIMENTS COMPLETE (with ${#FAILED_MODELS[@]} failure(s))"
      echo " Failed: ${FAILED_MODELS[*]}"
    fi
    echo " Results in: $RESULTS_DIR/"
    echo "============================================================"
    ;;
  reset)
    rm -f "$CHECKPOINT_FILE"
    echo "[*] Checkpoint cleared. All experiments will re-run."
    ;;
  *)
    echo "Usage: bash experiments/run_profiling.sh [phase]"
    echo ""
    echo "Phases:"
    echo "  all            Run every phase below in order (with checkpoint/resume)"
    echo "  gemma4         Profile Gemma 4 (GQA + local/global)"
    echo "  glm47flash     Profile GLM-4.7-Flash (MoE+DSA)"
    echo "  gptoss         Profile gpt-oss-120b (sliding/full hybrid + MoE)"
    echo "  deepseek       Profile DeepSeek (MLA) — only if weights present"
    echo "  nemotron       Profile Nemotron-H (Mamba+attn) — only if weights present"
    echo "  lfm25          Profile LFM 2.5 (Liquid Foundation Model)"
    echo "  qwen3          Profile Qwen3-32B (dense GQA) — only if weights present"
    echo "  qwen36         Profile Qwen3.6-27B (DeltaNet + Gated Attention hybrid)"
    echo "  llama          Profile Llama-3.1-70B (dense GQA) — only if weights present"
    echo "  llama4         Profile Llama-4-Scout (MoE) — only if weights present"
    echo "  mha_baseline   Profile Pythia-1.4B as the pure-MHA anchor"
    echo "  mitigations    Run quantization / H2O / prefix-sharing benchmark"
    echo "  comparative    Build the cross-architecture comparison JSON"
    echo "  perplexity     WikiText-103 perplexity for every available model"
    echo "  reset          Clear the checkpoint file (forces all phases to rerun)"
    ;;
esac
