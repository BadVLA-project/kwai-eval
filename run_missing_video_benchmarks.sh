#!/bin/bash
# ==========================================================================
# Run the remaining paper video benchmarks in this eval framework.
#
# Supported here:
#   - VCRBench
#   - CVBench cross-video reasoning
#   - LongVideoBench
#   - TempCompass
# ==========================================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${NGPU:-}" ]; then
  NGPU=$(python3 -c "
import subprocess
try:
    out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    print(len([line for line in out.splitlines() if line.strip()]))
except Exception:
    print(1)
")
fi

GPU_OFFSET="${GPU_OFFSET:-0}"
REUSE="${REUSE:-1}"
DELAY="${DELAY:-15}"
EVAL_ID_MODE="${EVAL_ID_MODE:-day}"
EVAL_ID="${EVAL_ID:-}"
WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_missing_video_benchmarks}"

export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SKIP_ERR="${SKIP_ERR:-1}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-102400}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

export USE_COT="${USE_COT:-0}"
export TEMPERATURE="${TEMPERATURE:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

if [ "${NGPU}" -ge 8 ]; then
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
  export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-32}"
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
  export VLLM_BUILD_WORKERS="${VLLM_BUILD_WORKERS:-2}"
else
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
  export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-32}"
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
  export VLLM_BUILD_WORKERS="${VLLM_BUILD_WORKERS:-4}"
fi
export VLLM_DECODE_TIMEOUT="${VLLM_DECODE_TIMEOUT:-900}"

if [ -n "${MODELS:-}" ]; then
  read -ra MODEL_LIST <<< "${MODELS}"
else
  MODEL_LIST=(
    Qwen3-VL-4B-Instruct
  )
fi

if [ -n "${DATA:-}" ]; then
  read -ra DATASETS <<< "${DATA}"
else
  DATASETS=(
    VCRBench_64frame_nopack
    CVBench_64frame
    TempCompass_64frame
    LongVideoBench_64frame
  )
fi

CMD=(
  python launch_workers.py
  --ngpu "${NGPU}"
  --gpu-offset "${GPU_OFFSET}"
  --delay "${DELAY}"
  --
  run.py
  --use-vllm
  --data "${DATASETS[@]}"
  --model "${MODEL_LIST[@]}"
  --work-dir "${WORK_DIR}"
)

if [ -n "${JUDGE:-}" ]; then
  CMD+=(--judge "${JUDGE}")
fi

if [ -n "${JUDGE_ARGS:-}" ]; then
  CMD+=(--judge-args "${JUDGE_ARGS}")
fi

if [ -n "${EVAL_ID}" ]; then
  CMD+=(--eval-id "${EVAL_ID}")
else
  CMD+=(--eval-id-mode "${EVAL_ID_MODE}")
fi

if [ "${REUSE}" = "1" ]; then
  CMD+=(--reuse)
fi

echo "=================================================================="
echo " [$(date '+%Y-%m-%d %H:%M:%S')] missing video benchmarks"
echo "   Datasets: ${DATASETS[*]}"
echo "   Models:   ${MODEL_LIST[*]}"
echo "   GPUs:     ${NGPU} (offset=${GPU_OFFSET})"
echo "   work_dir: ${WORK_DIR}"
echo "=================================================================="

"${CMD[@]}"
