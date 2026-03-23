#!/bin/bash
# ==========================================================================
# run_mlvu_test.sh — Debug MLVU_MCQ on 2 GPUs (no CoT, greedy)
#
# Usage:
#   NGPU=2 bash run_mlvu_test.sh
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
export SKIP_ERR="${SKIP_ERR:-0}"

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
export MLVU_DIR="${MLVU_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/MLVU_Test}"

# vLLM settings
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-32}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

# No CoT, greedy
export USE_COT=0
export TEMPERATURE=0

WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_mlvu_debug}"
LOG_FILE="${WORK_DIR}/mlvu_test.log"
mkdir -p "${WORK_DIR}"

echo "===== MLVU MCQ Debug Test (${NGPU} GPUs) ====="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

python launch_workers.py \
  --ngpu "${NGPU}" \
  --delay "${DELAY}" \
  -- \
  run.py \
  --use-vllm \
  --data MLVU_MCQ_64frame \
  --model Qwen3-VL-4B-Instruct \
  --work-dir "${WORK_DIR}" \
  2>&1 | tee "${LOG_FILE}"

rc=${PIPESTATUS[0]}
echo "Exit code: ${rc}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
exit ${rc}
