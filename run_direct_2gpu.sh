#!/bin/bash
# ==========================================================================
# run_direct_8gpu.sh — Direct answer (no CoT), GPU 0-7 (全8卡)
#
#   USE_COT=0:    greedy, temperature=0, model answers directly
#   GPU:          0,1,2,3,4,5,6,7  (全8卡)
#
# Usage (MUST run inside tmux/screen to survive SSH disconnection):
#   tmux new -s eval
#   bash run_direct_8gpu.sh 2>&1 | tee ${WORK_DIR:-/tmp}/run_direct_8gpu.log
#   # Ctrl+B D to detach, tmux attach -t eval to reattach
#
#   REUSE=1 bash run_direct_8gpu.sh   # skip completed model x dataset
#   MODEL=Qwen3-VL-7B bash run_direct_8gpu.sh
# ==========================================================================
set -euo pipefail
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
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-102400}"

REUSE="${REUSE:-0}"
DELAY="${DELAY:-15}"
EVAL_ID_MODE="${EVAL_ID_MODE:-day}"
EVAL_ID="${EVAL_ID:-}"

# ---------------------------------------------------------------------------
# Direct answer mode: USE_COT=0 → greedy, temperature=0
# ---------------------------------------------------------------------------
export USE_COT=0
export TEMPERATURE=0
export TIMELENS_EVAL_MODE=timelens

NGPU=2
GPU_OFFSET=0

# ---------------------------------------------------------------------------
# vLLM settings for 80GB GPUs
# 8-GPU note: reduce VLLM_BUILD_WORKERS to avoid CPU oversubscription
# ---------------------------------------------------------------------------
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export VLLM_BUILD_WORKERS="${VLLM_BUILD_WORKERS:-2}"

# ---------------------------------------------------------------------------
# HF / offline settings
# ---------------------------------------------------------------------------
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=0
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# ---------------------------------------------------------------------------
# Local dataset overrides
# ---------------------------------------------------------------------------
export VIDEO_HOLMES_DIR="${VIDEO_HOLMES_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/Video-Holmes}"
export TIMELENS_DIR="${TIMELENS_DIR:-/m2v_intern/xuboshen/zgw/hf_cache_temp/TimeLens-Bench}"
export PERCEPTION_TEST_DIR="${PERCEPTION_TEST_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/PerceptionTest}"
export MLVU_DIR="${MLVU_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/MLVU_Test}"
export ETBENCH_DIR="${ETBENCH_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/ETBench}"

# ---------------------------------------------------------------------------
# Datasets & Model
# ---------------------------------------------------------------------------
DATASETS=(
  AoTBench_ReverseFilm_adaptive
  AoTBench_UCF101_adaptive
  AoTBench_Rtime_t2v_adaptive
  AoTBench_Rtime_v2t_adaptive
  AoTBench_QA_adaptive
  FutureOmni_adaptive
  CharadesTimeLens_adaptive
  CharadesSTA_adaptive
  MVBench_MP4_adaptive
  PerceptionTest_val_adaptive
  Video_Holmes_adaptive
  Video-MME_adaptive
  ETBench_adaptive
  MLVU_MCQ_adaptive
  TimeLensBench_Charades_adaptive
  TimeLensBench_ActivityNet_adaptive
  TimeLensBench_QVHighlights_adaptive
)

MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"

MODELS=(
  Qwen3-VL-4B-Instruct_aot_v2t
)

WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_2gpu}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
CMD=(
  python launch_workers.py
  --ngpu "${NGPU}"
  --gpu-offset "${GPU_OFFSET}"
  --delay "${DELAY}"
  --
  run.py
  --use-vllm
  --judge exact_matching
  --data "${DATASETS[@]}"
  --model "${MODELS[@]}"
  --work-dir "${WORK_DIR}"
)

if [ -n "${EVAL_ID}" ]; then
  CMD+=(--eval-id "${EVAL_ID}")
else
  CMD+=(--eval-id-mode "${EVAL_ID_MODE}")
fi

if [ "${REUSE}" = "1" ]; then
  CMD+=(--reuse)
fi

echo "=================================================================="
echo " [$(date '+%Y-%m-%d %H:%M:%S')] run_direct_8gpu: USE_COT=0  TEMPERATURE=0"
echo "   GPUs: 0-7  (全8卡)"
echo "   work_dir: ${WORK_DIR}"
echo "   eval_id_mode: ${EVAL_ID_MODE}"
if [ -n "${EVAL_ID}" ]; then
  echo "   eval_id: ${EVAL_ID}"
fi
echo "=================================================================="

# ---------------------------------------------------------------------------
# GPU filler — keeps util high during idle gaps (barriers, prompt building, etc.)
# Disable with: GPU_FILLER=0
# ---------------------------------------------------------------------------
GPU_FILLER="${GPU_FILLER:-1}"
FILLER_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build GPU list from offset
FILLER_GPU_LIST=""
for i in $(seq 0 $((NGPU - 1))); do
  [ -n "${FILLER_GPU_LIST}" ] && FILLER_GPU_LIST="${FILLER_GPU_LIST},"
  FILLER_GPU_LIST="${FILLER_GPU_LIST}$((GPU_OFFSET + i))"
done

if [ "${GPU_FILLER}" = "1" ] && [ -f "${SCRIPT_DIR}/gpu_filler.py" ]; then
  echo "[eval] Starting GPU filler on GPUs ${FILLER_GPU_LIST} (target=${FILLER_TARGET_UTIL:-80}%) ..."
  python "${SCRIPT_DIR}/gpu_filler.py" \
    --gpus "${FILLER_GPU_LIST}" \
    --target-util "${FILLER_TARGET_UTIL:-80}" \
    --matrix-size "${FILLER_MATRIX_SIZE:-4096}" \
    --batch "${FILLER_BATCH:-10}" \
    --gap-matrix "${FILLER_GAP_MATRIX:-2048}" \
    --push-matrix "${FILLER_PUSH_MATRIX:-3072}" &
  FILLER_PID=$!
  echo "[eval] GPU filler started (PID=${FILLER_PID})"
fi

cleanup_filler() {
  if [ -n "${FILLER_PID}" ]; then
    kill "${FILLER_PID}" 2>/dev/null || true
    wait "${FILLER_PID}" 2>/dev/null || true
    echo "[eval] GPU filler stopped"
  fi
}
trap cleanup_filler EXIT

"${CMD[@]}"
