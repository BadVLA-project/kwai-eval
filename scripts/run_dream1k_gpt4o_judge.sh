#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_NAME [WORK_DIR]" >&2
  echo "Example: $0 Tarsier ./outputs" >&2
  exit 2
fi

MODEL_NAME="$1"
WORK_DIR="${2:-${WORK_DIR:-./outputs}}"
DATASET="${DATASET:-DREAM-1K_adaptive}"
API_NPROC="${API_NPROC:-4}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-16384}"
JUDGE_ARGS="${JUDGE_ARGS:-{\"use_azure_sdk\": true, \"max_completion_tokens\": ${MAX_COMPLETION_TOKENS}}}"

export DREAM1K_DIR="${DREAM1K_DIR:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DREAM-1K}"

if [[ -z "${AZURE_OPENAI_API_KEY:-${AZURE_API_KEY:-}}" ]]; then
  echo "Missing Azure key. Set AZURE_OPENAI_API_KEY or AZURE_API_KEY." >&2
  exit 2
fi

if [[ -z "${AZURE_OPENAI_ENDPOINT:-${ENDPOINT_URL:-${AZURE_ENDPOINT:-}}}" ]]; then
  echo "Missing Azure endpoint. Set ENDPOINT_URL, AZURE_OPENAI_ENDPOINT, or AZURE_ENDPOINT." >&2
  exit 2
fi

if [[ -z "${AZURE_OPENAI_DEPLOYMENT_NAME:-${DEPLOYMENT_NAME:-${AZURE_DEPLOYMENT_NAME:-}}}" ]]; then
  echo "Missing Azure deployment. Set DEPLOYMENT_NAME, AZURE_OPENAI_DEPLOYMENT_NAME, or AZURE_DEPLOYMENT_NAME." >&2
  exit 2
fi

if [[ -n "${PRED_FILE:-}" ]]; then
  python - <<PY
from vlmeval.dataset import build_dataset

dataset = build_dataset("${DATASET}")
result = dataset.evaluate(
    "${PRED_FILE}",
    model="gpt-4o",
    nproc=${API_NPROC},
    use_azure_sdk=True,
    max_completion_tokens=${MAX_COMPLETION_TOKENS},
)
print(result)
PY
  exit 0
fi

python run.py \
  --mode eval \
  --data "${DATASET}" \
  --model "${MODEL_NAME}" \
  --work-dir "${WORK_DIR}" \
  --reuse \
  --judge gpt-4o \
  --judge-args "${JUDGE_ARGS}" \
  --api-nproc "${API_NPROC}"
