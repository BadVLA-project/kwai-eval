#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARSIER_DIR="${TARSIER_DIR:-${REPO_DIR}/../tarsier}"
DREAM1K_DIR="${DREAM1K_DIR:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DREAM-1K}"
OUT_DIR="${OUT_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/tarsier_dream1k}"
LIMIT="${LIMIT:--1}"

if [[ -z "${TARSIER_MODEL:-}" ]]; then
  echo "Missing TARSIER_MODEL. Example: TARSIER_MODEL=/path/to/tarsier2 bash $0" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
ANN_FILE="${OUT_DIR}/DREAM-1k.local.jsonl"

python "${REPO_DIR}/scripts/prepare_tarsier_dream1k.py" \
  --tarsier-ann "${TARSIER_DIR}/data/annotations/DREAM-1k.jsonl" \
  --dream-root "${DREAM1K_DIR}" \
  --output "${ANN_FILE}" \
  --limit "${LIMIT}"

cd "${TARSIER_DIR}"
python3 -m tasks.inference_caption \
  --model_name_or_path "${TARSIER_MODEL}" \
  --config configs/tarser2_default_config.yaml \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  --top_p 1 \
  --temperature 0 \
  --input_file "${ANN_FILE}" \
  --output_dir "${OUT_DIR}" \
  --output_name predictions \
  --max_n_samples_per_benchmark -1 \
  --resume "${RESUME:-True}" \
  --num_chunks "${NUM_CHUNKS:-1}" \
  --chunk_idx "${CHUNK_IDX:-0}"
