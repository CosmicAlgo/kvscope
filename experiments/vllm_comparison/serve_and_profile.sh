#!/usr/bin/env bash
# =============================================================================
# serve_and_profile.sh — vLLM serving + black-box profiling for MiniMax M3
# =============================================================================
# Launches `vllm serve` for MiniMax M3 (or the M2 stand-in), waits for health,
# then drives the KVScope API behavioural profiler against the local endpoint
# while scraping vLLM's Prometheus KV-cache metrics. This is the production-
# realistic counterpart to the in-process tracer: vLLM PagedAttention is how M3
# is actually served, and it can reach the long contexts eager transformers
# cannot.
#
# Usage:
#   bash experiments/vllm_comparison/serve_and_profile.sh
#
# Environment:
#   MODEL_PATH      — HF id or local path (default: MiniMaxAI/MiniMax-M2 stand-in)
#   MODEL_NAME      — model name advertised by the server (default: basename)
#   TP              — tensor-parallel size (default: number of visible GPUs)
#   MAX_MODEL_LEN   — server max context (default: 131072)
#   PORT            — server port (default: 8000)
#   LENGTHS         — latency sweep input lengths (default: 1000,4000,16000,64000)
#   NEEDLE_LENGTHS  — needle context lengths (default: 16000,64000)
#   RESULTS_DIR     — output dir (default: ./results/api)
# =============================================================================

set -uo pipefail

MODEL_PATH="${MODEL_PATH:-MiniMaxAI/MiniMax-M2}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
LENGTHS="${LENGTHS:-1000,4000,16000,64000}"
NEEDLE_LENGTHS="${NEEDLE_LENGTHS:-16000,64000}"
RESULTS_DIR="${RESULTS_DIR:-./results/api}"

# Default tensor-parallel = number of visible GPUs (>=1).
if [[ -z "${TP:-}" ]]; then
  TP="$(python3 -c 'import torch;print(max(1,torch.cuda.device_count()))' 2>/dev/null || echo 1)"
fi

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo " vLLM serve + profile : $MODEL_NAME"
echo " model_path : $MODEL_PATH"
echo " TP / maxlen: $TP / $MAX_MODEL_LEN   port: $PORT"
echo " lengths    : $LENGTHS"
echo "============================================================"

if ! command -v vllm >/dev/null 2>&1; then
  echo "[!] vllm not installed. Install with: pip install vllm  (or: pip install -e '.[vllm]')"
  exit 1
fi

# ── Launch server ─────────────────────────────────────────────────────────────
echo "[*] Starting vLLM server..."
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --port "$PORT" >"$RESULTS_DIR/vllm_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "[*] Stopping vLLM server (pid $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Wait for health ─────────────────────────────────────────────────────────────
echo "[*] Waiting for server health (up to 600s)..."
for i in $(seq 1 600); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[*] Server healthy after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[!] Server process died. Last log lines:"
    tail -n 30 "$RESULTS_DIR/vllm_server.log"
    exit 1
  fi
  sleep 1
done

export KVSCOPE_API_KEY="${KVSCOPE_API_KEY:-EMPTY}"   # local vLLM ignores the key
BASE_URL="http://localhost:$PORT/v1"

# ── Scrape KV metrics in the background during profiling ───────────────────────
python3 experiments/vllm_comparison/scrape_vllm_metrics.py \
  --url "http://localhost:$PORT/metrics" --interval 1 --duration 240 \
  --output "$RESULTS_DIR/vllm_${MODEL_NAME}_kv_timeseries.json" &
METRICS_PID=$!

# ── Latency sweep (prefill + decode scaling) ───────────────────────────────────
echo "[*] Running latency sweep..."
python3 -m src.cli api --provider openai-compatible --base-url "$BASE_URL" --model "$MODEL_NAME" \
  latency --lengths "$LENGTHS" --max-tokens 128 \
  -o "$RESULTS_DIR/vllm_${MODEL_NAME}_latency.json"

# ── Needle-in-a-haystack (effective context) ───────────────────────────────────
echo "[*] Running needle recall grid..."
python3 -m src.cli api --provider openai-compatible --base-url "$BASE_URL" --model "$MODEL_NAME" \
  needle --lengths "$NEEDLE_LENGTHS" --depths 0,0.25,0.5,0.75,1.0 \
  -o "$RESULTS_DIR/vllm_${MODEL_NAME}_needle.json"

wait "$METRICS_PID" 2>/dev/null || true

echo ""
echo "============================================================"
echo " DONE. Artifacts in $RESULTS_DIR/:"
echo "   vllm_${MODEL_NAME}_latency.json       (TTFT/ITL scaling)"
echo "   vllm_${MODEL_NAME}_needle.json        (effective-context recall)"
echo "   vllm_${MODEL_NAME}_kv_timeseries.json (server KV-cache usage)"
echo "============================================================"
