#!/usr/bin/env bash
# =============================================================================
# run_spec_decode.sh — Speculative Decoding KV Cache Profiling
# =============================================================================
# Llama-3.1-70B (verifier) + Llama-3.2-1B (draft) speculative decoding.
# This is the fallback for Llama-4-Scout when it OOMs or produces < 5/20 valid.
#
# Usage:
#   bash experiments/run_spec_decode.sh
# =============================================================================

set -uo pipefail

MODELS_DIR="${MODELS_DIR:-/root/models}"
RESULTS_DIR="${RESULTS_DIR:-./results_v2}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

VERIFIER_PATH="${MODELS_DIR}/llama31-70b"
DRAFT_PATH="${MODELS_DIR}/llama32-1b"
OUTPUT="${RESULTS_DIR}/spec_decode_profile.json"
PROMPTS_FILE="${RESULTS_DIR}/prompts.json"

echo "============================================================"
echo " KVScope — Speculative Decoding Profiling"
echo "============================================================"
echo " Verifier : Llama-3.1-70B ($VERIFIER_PATH)"
echo " Draft    : Llama-3.2-1B ($DRAFT_PATH)"
echo " Tokens   : $MAX_TOKENS"
echo " Output   : $OUTPUT"
echo "============================================================"

# Check models exist
if [[ ! -d "$VERIFIER_PATH" ]]; then
  echo "[!] Verifier not found: $VERIFIER_PATH"
  echo "    Download: huggingface-cli download meta-llama/Llama-3.1-70B --local-dir $VERIFIER_PATH"
  exit 1
fi

if [[ ! -d "$DRAFT_PATH" ]]; then
  echo "[!] Draft not found: $DRAFT_PATH"
  echo "    Download: huggingface-cli download meta-llama/Llama-3.2-1B --local-dir $DRAFT_PATH"
  exit 1
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
  cp experiments/configs/prompts_advanced.json "$PROMPTS_FILE"
fi

python3 -m src.models.spec_decode_runner \
  --verifier-path "$VERIFIER_PATH" \
  --draft-path "$DRAFT_PATH" \
  --prompts-file "$PROMPTS_FILE" \
  --max-tokens "$MAX_TOKENS" \
  --output "$OUTPUT"

echo ""
echo "[✓] Speculative decoding profiling complete: $OUTPUT"
