#!/usr/bin/env bash
# Run Qwen over the complete SCOPE-Bench item catalog.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_ITEMS="${INPUT_ITEMS:-${REPO_ROOT}/datasets/ShortVideoFull/items_final_fixed.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${SCRIPT_DIR}/results/Qwen3_7_Max_full_t0p3_seed42_scores.jsonl}"
OUTPUT_CSV="${OUTPUT_CSV:-${SCRIPT_DIR}/results/Qwen3_7_Max_full_t0p3_seed42_scores.csv}"
MODEL="${MODEL:-qwen/qwen3.7-max}"
CONCURRENCY="${CONCURRENCY:-30}"
TEMPERATURE="${TEMPERATURE:-0.3}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
MAX_RETRIES="${MAX_RETRIES:-6}"
SESSION_ID="${SESSION_ID:-scope-bench-qwen-full-v1}"

if [[ ! -f "${INPUT_ITEMS}" ]]; then
  echo "Missing input catalog: ${INPUT_ITEMS}" >&2
  echo "Download or prepare ShortVideoFull first; see datasets/README.md." >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" && -z "${API_KEY:-}" ]]; then
  echo "Set OPENROUTER_API_KEY (recommended) or API_KEY before running." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSONL}")" "$(dirname "${OUTPUT_CSV}")"

extra_args=()
if [[ -n "${LIMIT:-}" ]]; then
  extra_args+=(--limit "${LIMIT}")
fi
if [[ "${NO_PROMPT_CACHE:-0}" == "1" ]]; then
  extra_args+=(--no_prompt_cache)
fi

status=0
"${PYTHON_BIN}" evaluate_videos.py \
  --input "${INPUT_ITEMS}" \
  --output "${OUTPUT_JSONL}" \
  --provider openrouter \
  --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" \
  --temperature "${TEMPERATURE}" \
  --seed "${SEED}" \
  --max_tokens "${MAX_TOKENS}" \
  --request_timeout "${REQUEST_TIMEOUT}" \
  --max_retries "${MAX_RETRIES}" \
  --session_id "${SESSION_ID}" \
  --continue_on_preflight_failure \
  "${extra_args[@]}" || status=$?

# Always refresh the readable CSV, including after a partially successful run.
"${PYTHON_BIN}" scores_jsonl_to_csv.py \
  --input "${OUTPUT_JSONL}" \
  --output "${OUTPUT_CSV}" \
  --items "${INPUT_ITEMS}"

exit "${status}"
