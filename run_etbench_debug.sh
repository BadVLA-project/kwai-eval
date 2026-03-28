#!/bin/bash
# ==========================================================================
# run_etbench_debug.sh — Debug ETBench_subset (470 samples) on 2 GPUs
#
# Usage:
#   bash run_etbench_debug.sh
#   CLEAN=0 bash run_etbench_debug.sh          # skip cache cleaning (reuse)
#   DATASET=ETBench_subset_1fps bash run_etbench_debug.sh
# ==========================================================================
set -uo pipefail
set -x

export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export VLLM_HOST_IP=127.0.0.1

export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

unset CC
unset CXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SKIP_ERR="${SKIP_ERR:-1}"

# Give decord more EOF retries for problematic videos (default 10240)
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-102400}"

NGPU="${NGPU:-2}"
DELAY="${DELAY:-15}"

# HF / offline settings
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=0
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# Local dataset override
export ETBENCH_DIR="${ETBENCH_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/ETBench}"

# Force decord backend (torchcodec has FFmpeg compatibility issues on this server)
export FORCE_QWENVL_VIDEO_READER=decord

# vLLM settings
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-32}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

# No CoT, greedy
export USE_COT="${USE_COT:-0}"
export TEMPERATURE="${TEMPERATURE:-0}"

DATASET="${DATASET:-ETBench_subset_1fps}"
MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"
WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_etbench_debug}"
LOG_FILE="${WORK_DIR}/etbench_debug.log"
mkdir -p "${WORK_DIR}"

# Clean stale cache (default: on)
CLEAN="${CLEAN:-1}"
if [ "${CLEAN}" = "1" ]; then
  echo "Cleaning stale ETBench cache in ${WORK_DIR} ..."
  find "${WORK_DIR}" -maxdepth 2 -name '*ETBench*' -type f \
    \( -name '*.pkl' -o -name '*.tsv' -o -name '*.jsonl' -o -name '*.xlsx' \
       -o -name '*etbench_score*' -o -name '*etbench_acc*' \) \
    -print -delete 2>/dev/null || true
  echo "Cache cleaned."
fi

echo "===== ETBench Subset Debug (${NGPU} GPUs) ====="
echo "Dataset:    ${DATASET}"
echo "Model:      ${MODEL}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

python launch_workers.py \
  --ngpu "${NGPU}" \
  --delay "${DELAY}" \
  -- \
  run.py \
  --use-vllm \
  --data "${DATASET}" \
  --model "${MODEL}" \
  --work-dir "${WORK_DIR}" \
  2>&1 | tee "${LOG_FILE}"

rc=${PIPESTATUS[0]}
echo "Exit code: ${rc}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
exit ${rc}
