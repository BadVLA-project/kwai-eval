#!/bin/bash
# ==========================================================================
# run_videossr_64frame.sh — VideoSSR-8B evaluation with 64-frame settings.
#
# VideoSSR reports its No-CoT greedy results under fixed-frame settings
# including 64 frames. This launcher maps the requested *_adaptive benchmark
# list to the corresponding *_64frame dataset registrations in this repo.
#
# Usage:
#   bash run_videossr_64frame.sh
#   NGPU=8 REUSE=1 bash run_videossr_64frame.sh
#   DATA="Video-MME_64frame LongVideoBench_64frame" bash run_videossr_64frame.sh
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
  echo "[auto-detect] Found ${NGPU} GPU(s)"
fi

GPU_OFFSET="${GPU_OFFSET:-0}"
REUSE="${REUSE:-1}"
DELAY="${DELAY:-15}"
EVAL_ID_MODE="${EVAL_ID_MODE:-day}"
EVAL_ID="${EVAL_ID:-}"
WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_videossr_64frame}"
export VIDEO_SSR_FRAMES="${VIDEO_SSR_FRAMES:-64}"

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
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-102400}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# VideoSSR paper setting: direct answer / greedy decoding.
export USE_COT="${USE_COT:-0}"
export TEMPERATURE="${TEMPERATURE:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# Local dataset overrides used by these benchmarks.
export VIDEO_HOLMES_DIR="${VIDEO_HOLMES_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/Video-Holmes}"
export VIDEO_MME_DIR="${VIDEO_MME_DIR:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--lmms-lab--Video-MME}"
export TIMELENS_DIR="${TIMELENS_DIR:-/m2v_intern/xuboshen/zgw/hf_cache_temp/TimeLens-Bench}"
export MLVU_DIR="${MLVU_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/MLVU_Test}"
export ETBENCH_DIR="${ETBENCH_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/ETBench}"

if [ "${NGPU}" -ge 8 ]; then
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
  export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-32}"
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
  export VLLM_BUILD_WORKERS="${VLLM_BUILD_WORKERS:-2}"
elif [ "${NGPU}" -ge 4 ]; then
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
  export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-32}"
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
  export VLLM_BUILD_WORKERS="${VLLM_BUILD_WORKERS:-4}"
else
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
  export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-16}"
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
  export VLLM_BUILD_WORKERS="${VLLM_BUILD_WORKERS:-4}"
fi
export VLLM_DECODE_TIMEOUT="${VLLM_DECODE_TIMEOUT:-900}"

if [ "${VIDEO_SSR_FRAMES}" != "64" ]; then
  echo "ERROR: this launcher is pinned to VideoSSR 64-frame datasets. Got VIDEO_SSR_FRAMES=${VIDEO_SSR_FRAMES}" >&2
  exit 1
fi

if [ -n "${DATA:-}" ]; then
  read -ra DATASETS <<< "${DATA}"
else
  DATASETS=(
    TimeLensBench_QVHighlights_64frame
    TimeLensBench_ActivityNet_64frame
    TimeLensBench_Charades_64frame
    Video_Holmes_64frame
    VideoMMMU_64frame
    Video_TT_64frame
    ETBench_64frame
    MVBench_MP4_64frame
    TempCompass_MCQ_64frame
    AoTBench_ReverseFilm_64frame
    AoTBench_UCF101_64frame
    AoTBench_Rtime_t2v_64frame
    AoTBench_Rtime_v2t_64frame
    AoTBench_QA_64frame
    Vinoground_64frame
    Video-MME_64frame
    MLVU_MCQ_64frame
    LongVideoBench_64frame
  )
fi

if [ -n "${MODELS:-}" ]; then
  read -ra MODEL_LIST <<< "${MODELS}"
else
  MODEL_LIST=(
    VideoSSR-8B
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
echo " [$(date '+%Y-%m-%d %H:%M:%S')] VideoSSR 64-frame evaluation"
echo "   GPUs:        ${NGPU} (offset=${GPU_OFFSET})"
echo "   Models:      ${MODEL_LIST[*]}"
echo "   Datasets:    ${#DATASETS[@]} benchmarks"
echo "   Frames:      ${VIDEO_SSR_FRAMES}"
echo "   USE_COT:     ${USE_COT}"
echo "   TEMPERATURE: ${TEMPERATURE}"
echo "   work_dir:    ${WORK_DIR}"
echo "   reuse:       ${REUSE}"
if [ -n "${EVAL_ID}" ]; then
  echo "   eval_id:     ${EVAL_ID}"
else
  echo "   eval_id_mode:${EVAL_ID_MODE}"
fi
echo "=================================================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
