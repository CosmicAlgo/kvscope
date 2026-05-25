#!/usr/bin/env bash
# =============================================================================
# run_longctx_sweep.sh — Long-Context KV Cache Growth Sweep
# =============================================================================
# Runs Llama-3.1-70B at multiple MAX_TOKENS values to observe non-linear
# KV cache growth, fragmentation onset, and sliding-window saturation.
#
# Output: results_v2/longctx_sweep/llama_<N>tok_profile.json
#
# Usage:
#   bash experiments/run_longctx_sweep.sh
#
# Environment:
#   MODELS_DIR    — path to models (default: /root/models)
#   RESULTS_DIR   — base results dir (default: ./results_v2)
# =============================================================================

set -uo pipefail

MODELS_DIR="${MODELS_DIR:-/root/models}"
RESULTS_DIR="${RESULTS_DIR:-./results_v2}"
SWEEP_DIR="${RESULTS_DIR}/longctx_sweep"
CAPTURE_EVERY="${CAPTURE_EVERY:-5}"
MODEL_PATH="${MODELS_DIR}/llama31-70b"
PROMPTS_FILE="${RESULTS_DIR}/prompts.json"

# Token lengths to sweep
TOKEN_LENGTHS=(512 2048 8192 32768)

mkdir -p "$SWEEP_DIR"

echo "============================================================"
echo " KVScope — Long-Context KV Growth Sweep"
echo "============================================================"
echo " Model    : Llama-3.1-70B"
echo " Path     : $MODEL_PATH"
echo " Lengths  : ${TOKEN_LENGTHS[*]}"
echo " Output   : $SWEEP_DIR/"
echo "============================================================"
echo ""

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[!] Model not found at $MODEL_PATH"
  echo "    Download: huggingface-cli download meta-llama/Llama-3.1-70B --local-dir $MODEL_PATH"
  exit 1
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
  cp experiments/configs/prompts_advanced.json "$PROMPTS_FILE"
fi

FAILED=()

for N in "${TOKEN_LENGTHS[@]}"; do
  OUTPUT="${SWEEP_DIR}/llama_${N}tok_profile.json"

  if [[ -f "$OUTPUT" ]]; then
    echo "[SKIP] $OUTPUT already exists"
    continue
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " Sweeping: MAX_TOKENS=$N"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if ! python3 -m src.models.llama_runner \
    --model-path "$MODEL_PATH" \
    --max-tokens "$N" \
    --capture-every "$CAPTURE_EVERY" \
    --output "$OUTPUT" \
    --prompts-file "$PROMPTS_FILE" \
    --load-in-4bit; then
    echo "[!] FAILED at MAX_TOKENS=$N"
    FAILED+=("$N")
  fi
done

echo ""
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo " LONG-CONTEXT SWEEP COMPLETE"
else
  echo " SWEEP COMPLETE (${#FAILED[@]} failures: ${FAILED[*]})"
fi
echo " Results: $SWEEP_DIR/"
echo "============================================================"
ls -la "$SWEEP_DIR/"
