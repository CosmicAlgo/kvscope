#!/bin/bash
# =============================================================================
# run_full_collection.sh — KVScope: one-shot data collection + bundle
# =============================================================================
# Profiles all working models with the advanced prompt set, generates plots,
# writes a paper-ready Markdown summary, and packages everything into a single
# tar.gz archive that can be scp'd home before the GPU droplet is destroyed.
#
# Usage:
#   bash experiments/run_full_collection.sh             # default: 4096 tokens × 15 prompts
#   MAX_TOKENS=2048 bash experiments/run_full_collection.sh
#   MODELS="gemma4 gptoss" bash experiments/run_full_collection.sh
#   PROMPTS_FILE=path/to/custom.json bash experiments/run_full_collection.sh
#
# Environment variables:
#   MAX_TOKENS       — generation length per prompt (default: 4096)
#   MODELS           — space-separated model list (default: "gemma4 glm51 gptoss")
#   PROMPTS_FILE     — path to JSON array of prompts
#                      (default: experiments/configs/prompts_advanced.json)
#   RESULTS_DIR      — base results directory (default: ./results)
#   CAPTURE_EVERY    — sample every Nth decode step (default: 1; bump for long runs)
#   SKIP_BUNDLE      — set to 1 to skip the final tar step
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
RESULTS_DIR="${RESULTS_DIR:-./results}"
RUN_DIR="${RESULTS_DIR}/full_run_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
PLOTS_DIR="${RUN_DIR}/plots"

MAX_TOKENS="${MAX_TOKENS:-4096}"
# Sampling cadence: at 4096 tokens × 36 layers × 15 prompts, capturing every
# step would produce ~400 MB JSONs per model. CAPTURE_EVERY=8 still gives 512
# data points per prompt — plenty for fitting growth curves and detector stats.
CAPTURE_EVERY="${CAPTURE_EVERY:-8}"
# Default model set now includes the pure-MHA baseline (Pythia-1.4B) and Nemotron.
# The baseline gets capped at 1024 tokens automatically by run_profiling.sh
# because Pythia max_position_embeddings=2048.
MODELS="${MODELS:-mha_baseline gemma4 glm51 gptoss nemotron lfm25 deepseek_v4}"
PROMPTS_FILE="${PROMPTS_FILE:-experiments/configs/prompts_advanced.json}"
# Whether to also run WikiText-103 perplexity at the end (1=yes). This adds
# ~30 min on top of profiling but produces the publication-grade quality
# numbers.
RUN_PERPLEXITY="${RUN_PERPLEXITY:-1}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$PLOTS_DIR"
MASTER_LOG="${LOG_DIR}/full_run.log"

# Mirror all output to the master log
exec > >(tee -a "$MASTER_LOG") 2>&1

# Activate venv if present
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
  echo "[*] Using venv: $REPO_ROOT/.venv"
fi

PYTHON="${PYTHON:-python3}"

# ─── Helpers ──────────────────────────────────────────────────────────────────

banner() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════"
  echo " $1"
  echo "═══════════════════════════════════════════════════════════════════════"
}

step() { echo ""; echo "───── [$1] $2"; }

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[!] Missing required file: $1" >&2
    exit 1
  fi
}

# ─── Banner ───────────────────────────────────────────────────────────────────

banner "KVScope Full Collection — ${TIMESTAMP}"
echo "  Models       : $MODELS"
echo "  Max tokens   : $MAX_TOKENS"
echo "  Capture every: $CAPTURE_EVERY step(s)"
echo "  Prompts file : $PROMPTS_FILE"
echo "  Run dir      : $RUN_DIR"
echo "  Master log   : $MASTER_LOG"

# ─── Step 1 — Pre-flight ──────────────────────────────────────────────────────

step "1/7" "Pre-flight checks"
require_file "$PROMPTS_FILE"

if ! command -v nvidia-smi >/dev/null; then
  echo "[!] nvidia-smi not found — are we on a GPU host?" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | sed 's/^/  GPU: /'

$PYTHON - <<'PY'
import sys
import torch
print(f"  Python: {sys.version.split()[0]}")
print(f"  Torch : {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

df -h / | tail -2 | sed 's/^/  /'
echo "  Free RAM:"
free -h | head -2 | sed 's/^/    /'

# Validate the prompts file
N_PROMPTS=$($PYTHON - <<PY
import json
with open("$PROMPTS_FILE") as f:
    p = json.load(f)
assert isinstance(p, list) and all(isinstance(x, str) for x in p), "Bad prompts format"
print(len(p))
PY
)
echo "  Prompts loaded: $N_PROMPTS"

# ─── Step 2 — Stage prompts and reset checkpoints ─────────────────────────────

step "2/7" "Stage prompts + reset checkpoints"

# run_profiling.sh reads $RESULTS_DIR/prompts.json (auto-generates if missing).
# Stage the advanced set into the active RESULTS_DIR so all phase scripts use it.
mkdir -p "$RESULTS_DIR"
cp "$PROMPTS_FILE" "$RESULTS_DIR/prompts.json"
echo "  Staged $PROMPTS_FILE -> $RESULTS_DIR/prompts.json"

# Clear checkpoints for the requested models so they re-run.
# Phase tags differ slightly: most use "<model>_profile" but the MHA baseline
# uses "mha_baseline_profile" and perplexity uses "perplexity_eval".
CKPT="$RESULTS_DIR/.checkpoint"
touch "$CKPT"
for m in $MODELS; do
  case "$m" in
    mha_baseline|mha) sed -i "/^mha_baseline_profile$/d" "$CKPT" || true ;;
    *)                sed -i "/^${m}_profile$/d" "$CKPT" || true ;;
  esac
done
sed -i "/^comparative_analysis$/d" "$CKPT" || true
if [[ "${RUN_PERPLEXITY:-1}" == "1" ]]; then
  sed -i "/^perplexity_eval$/d" "$CKPT" || true
fi
echo "  Cleared checkpoints for: $MODELS + comparative_analysis$([[ \"${RUN_PERPLEXITY:-1}\" == \"1\" ]] && echo \" + perplexity_eval\")"
echo "  Remaining .checkpoint contents:"
sed 's/^/    /' "$CKPT" || echo "    (empty)"

# ─── Step 3 — Run profiling for each model ────────────────────────────────────

step "3/7" "Profiling models"

export MAX_TOKENS CAPTURE_EVERY RESULTS_DIR

for m in $MODELS; do
  banner "Profiling: $m"
  PHASE_LOG="${LOG_DIR}/profile_${m}.log"
  if bash experiments/run_profiling.sh "$m" 2>&1 | tee "$PHASE_LOG"; then
    echo "  [OK] $m profile complete -> $RESULTS_DIR/${m}_profile.json"
  else
    echo "  [WARN] $m profiling exited non-zero; continuing."
  fi
done

# ─── Step 4 — Comparative analysis + (optional) perplexity ───────────────────

step "4a/7" "Comparative analysis"
bash experiments/run_profiling.sh comparative 2>&1 \
  | tee "${LOG_DIR}/comparative.log"

if [[ "${RUN_PERPLEXITY:-1}" == "1" ]]; then
  step "4b/7" "WikiText-103 perplexity (real quality metric)"
  bash experiments/run_profiling.sh perplexity 2>&1 \
    | tee "${LOG_DIR}/perplexity.log" || true
else
  echo "  RUN_PERPLEXITY=0; skipping perplexity evaluation."
fi

# ─── Step 5 — Snapshot results into the run directory ─────────────────────────

step "5/7" "Snapshot artifacts -> $RUN_DIR"

# Resolve the JSON filename for each model. Most use ${m}_profile.json,
# the MHA baseline writes mha_baseline_profile.json regardless of alias.
for m in $MODELS; do
  case "$m" in
    mha_baseline|mha) src="$RESULTS_DIR/mha_baseline_profile.json" ;;
    *)                src="$RESULTS_DIR/${m}_profile.json" ;;
  esac
  if [[ -f "$src" ]]; then
    cp -f "$src" "$RUN_DIR/"
    echo "  + $(basename "$src") ($(du -h "$src" | cut -f1))"
  fi
done
if [[ -f "$RESULTS_DIR/comparative_analysis.json" ]]; then
  cp -f "$RESULTS_DIR/comparative_analysis.json" "$RUN_DIR/"
  echo "  + comparative_analysis.json"
fi
if [[ -f "$RESULTS_DIR/perplexity.json" ]]; then
  cp -f "$RESULTS_DIR/perplexity.json" "$RUN_DIR/"
  echo "  + perplexity.json"
fi
cp -f "$RESULTS_DIR/prompts.json" "$RUN_DIR/prompts.json" 2>/dev/null || true

# Copy the run-level logs and any env snapshots from the master logs dir
if [[ -d "$RESULTS_DIR/logs" ]]; then
  # Copy only the latest few files to keep the bundle reasonably sized
  ls -1t "$RESULTS_DIR/logs/" 2>/dev/null | head -10 | while read -r f; do
    cp -f "$RESULTS_DIR/logs/$f" "$LOG_DIR/" 2>/dev/null || true
  done
fi

# ─── Step 6 — Plots and Markdown summary ──────────────────────────────────────

step "6/7" "Generate plots + SUMMARY.md"

if $PYTHON -c "import matplotlib" 2>/dev/null; then
  $PYTHON experiments/generate_plots.py \
    --results-dir "$RUN_DIR" \
    --out-dir "$PLOTS_DIR" 2>&1 | sed 's/^/  /' || true
else
  echo "  [WARN] matplotlib not installed; skipping plots."
  echo "         (pip install matplotlib to enable)"
fi

$PYTHON experiments/generate_summary.py \
  --results-dir "$RUN_DIR" \
  --out "$RUN_DIR/SUMMARY.md" \
  --repo-root "$REPO_ROOT" 2>&1 | sed 's/^/  /'

# ─── Step 7 — Tar bundle ──────────────────────────────────────────────────────

step "7/7" "Bundle"

if [[ "${SKIP_BUNDLE:-0}" == "1" ]]; then
  echo "  SKIP_BUNDLE=1; not creating tarball."
  BUNDLE_PATH=""
else
  BUNDLE_NAME="kvscope_${TIMESTAMP}.tar.gz"
  BUNDLE_PATH="$RESULTS_DIR/$BUNDLE_NAME"
  ( cd "$RESULTS_DIR" && tar czf "$BUNDLE_NAME" "full_run_${TIMESTAMP}" )
  BUNDLE_SIZE=$(du -h "$BUNDLE_PATH" | cut -f1)
  echo "  Wrote: $BUNDLE_PATH  ($BUNDLE_SIZE)"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

banner "Collection complete"
echo "  Run directory : $RUN_DIR"
echo "  Summary       : $RUN_DIR/SUMMARY.md"
echo "  Plots         : $PLOTS_DIR"
if [[ -n "$BUNDLE_PATH" ]]; then
  echo "  Bundle        : $BUNDLE_PATH"
  echo ""
  echo "  Download from your laptop:"
  echo "    scp root@<droplet-ip>:$REPO_ROOT/$BUNDLE_PATH ./"
  echo ""
  echo "  Or if you have HF_TOKEN set, push to HuggingFace Datasets:"
  echo "    huggingface-cli upload <user>/kv-cache-dynamics-data \\"
  echo "        $BUNDLE_PATH --repo-type dataset"
fi
echo ""
echo "  When you're done, delete the droplet:"
echo "    doctl compute droplet delete <droplet-id>"
echo ""
