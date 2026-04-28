#!/bin/bash
# ==========================================================================
# run_direct.sh — Unified direct answer evaluation (auto-detect GPUs)
#
# Features:
#   - Auto-detects available GPU count (override with NGPU=N)
#   - Auto-scales vLLM params based on GPU count
#   - Integrated NaN retry: RETRY_NAN=1 to re-run failed samples
#   - All datasets and models in one script
#
# Usage:
#   # Basic: auto-detect GPUs, run all datasets with default model
#   bash run_direct.sh
#
#   # Specify GPU count manually
#   NGPU=4 bash run_direct.sh
#
#   # Specify model(s) — space-separated in env var
#   MODELS="Qwen3-VL-4B-Instruct Qwen3-VL-4B-Instruct_aot_v2t" bash run_direct.sh
#
#   # Retry NaN predictions from a previous run
#   RETRY_NAN=1 bash run_direct.sh
#
#   # Reuse existing inference results (only re-evaluate)
#   REUSE=1 bash run_direct.sh
#
#   # Use with tmux (recommended for long runs)
#   tmux new -s eval
#   bash run_direct.sh 2>&1 | tee /tmp/eval_direct.log

#  DATA="ETBench_adaptive" bash run_direct.sh
# ==========================================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===========================================================================
# 1. Auto-detect GPU count
# ===========================================================================
if [ -z "${NGPU:-}" ]; then
  NGPU=$(python3 -c "
import subprocess, os
try:
    out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    print(len([l for l in out.strip().split('\n') if l.strip()]))
except Exception:
    print(1)
")
  echo "[auto-detect] Found ${NGPU} GPU(s)"
fi
GPU_OFFSET="${GPU_OFFSET:-0}"

# ===========================================================================
# 2. Common environment
# ===========================================================================
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

# Direct answer mode: USE_COT=0 → greedy, temperature=0
export USE_COT="${USE_COT:-0}"
export TEMPERATURE="${TEMPERATURE:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export TIMELENS_EVAL_MODE="${TIMELENS_EVAL_MODE:-timelens}"

# ===========================================================================
# 3. Auto-scale vLLM params based on GPU count
# ===========================================================================
# More GPUs → smaller batches, less memory per GPU, fewer decode workers
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
  # 1-2 GPUs: can afford larger batches
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
  export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-32}"
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
  export VLLM_BUILD_WORKERS="${VLLM_BUILD_WORKERS:-4}"
fi

# Decode timeout (increase for long videos on slow storage)
export VLLM_DECODE_TIMEOUT="${VLLM_DECODE_TIMEOUT:-900}"

# ===========================================================================
# 4. HF / offline settings
# ===========================================================================
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=0
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# ===========================================================================
# 5. Local dataset overrides
# ===========================================================================
export VIDEO_HOLMES_DIR="${VIDEO_HOLMES_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/Video-Holmes}"
export VIDEO_MME_DIR="${VIDEO_MME_DIR:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--lmms-lab--Video-MME}"
export TIMELENS_DIR="${TIMELENS_DIR:-/m2v_intern/xuboshen/zgw/hf_cache_temp/TimeLens-Bench}"
export PERCEPTION_TEST_DIR="${PERCEPTION_TEST_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/PerceptionTest}"
export MLVU_DIR="${MLVU_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/MLVU_Test}"
export ETBENCH_DIR="${ETBENCH_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/ETBench}"
export DREAM1K_DIR="${DREAM1K_DIR:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DREAM-1K}"

# ===========================================================================
# 6. Datasets (override with DATA="ds1 ds2" env var)
# ===========================================================================
if [ -n "${DATA:-}" ]; then
  read -ra DATASETS <<< "${DATA}"
else
  DATASETS=(
    TimeLensBench_Charades_adaptive
    TimeLensBench_ActivityNet_adaptive
    TimeLensBench_QVHighlights_adaptive
    Vinoground_adaptive
    Video-TT_adaptive
    AoTBench_ReverseFilm_adaptive
    AoTBench_UCF101_adaptive
    AoTBench_Rtime_t2v_adaptive
    AoTBench_Rtime_v2t_adaptive
    AoTBench_QA_adaptive
    MVBench_MP4_adaptive
    Video_Holmes_adaptive
    Video-MME_adaptive
    ETBench_adaptive
    MLVU_MCQ_adaptive
  )
fi

# ===========================================================================
# 7. Models (override with MODELS env var, or edit the list below)
# ===========================================================================
if [ -n "${MODELS:-}" ]; then
  # Parse space-separated MODELS env var into array
  read -ra MODEL_LIST <<< "${MODELS}"
else
  MODEL_LIST=(
    Qwen3-VL-4B-Instruct
  )
fi

# ===========================================================================
# 8. Work directory and run settings
# ===========================================================================
WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_final}"
REUSE="${REUSE:-0}"
RETRY_NAN="${RETRY_NAN:-0}"
DELAY="${DELAY:-15}"
EVAL_ID_MODE="${EVAL_ID_MODE:-day}"
EVAL_ID="${EVAL_ID:-}"
JUDGE="${JUDGE:-}"
JUDGE_ARGS="${JUDGE_ARGS:-}"

# ===========================================================================
# 9. NaN retry: rebuild pkl files so failed samples get re-attempted
# ===========================================================================
if [ "${RETRY_NAN}" = "1" ]; then
  export RETRY_EMPTY=1
  echo "=================================================================="
  echo " [NaN Retry] Scanning for failed predictions..."
  echo "=================================================================="

  # Find the latest eval directory for each model
  for m in "${MODEL_LIST[@]}"; do
    model_dir="${WORK_DIR}/${m}"
    if [ ! -d "${model_dir}" ]; then
      echo "  [${m}] No results directory found, skipping"
      continue
    fi
    # Find latest T* directory
    latest_eval=$(ls -d "${model_dir}"/T* 2>/dev/null | sort | tail -1)
    if [ -z "${latest_eval}" ]; then
      echo "  [${m}] No T* eval directories found, skipping"
      continue
    fi
    echo "  [${m}] Retrying NaN in: ${latest_eval}"
    python "${SCRIPT_DIR}/scripts/retry_nan.py" "${latest_eval}" --clean-scores || true
  done
  echo ""
fi

# ===========================================================================
# 10. Build and run command
# ===========================================================================
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

if [ -n "${JUDGE}" ]; then
  CMD+=(--judge "${JUDGE}")
fi

if [ -n "${JUDGE_ARGS}" ]; then
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
echo " [$(date '+%Y-%m-%d %H:%M:%S')] run_direct.sh"
echo "   GPUs:       ${NGPU} (offset=${GPU_OFFSET})"
echo "   Models:     ${MODEL_LIST[*]}"
echo "   Datasets:   ${#DATASETS[@]} benchmarks"
echo "   USE_COT:    ${USE_COT}"
echo "   TEMPERATURE:${TEMPERATURE}"
echo "   work_dir:   ${WORK_DIR}"
echo "   decode_timeout: ${VLLM_DECODE_TIMEOUT}s"
echo "   build_workers:  ${VLLM_BUILD_WORKERS}"
if [ -n "${EVAL_ID}" ]; then
  echo "   eval_id:    ${EVAL_ID}"
else
  echo "   eval_id_mode: ${EVAL_ID_MODE}"
fi
if [ "${RETRY_NAN}" = "1" ]; then
  echo "   retry_nan:  ON (RETRY_EMPTY=1)"
fi
echo "=================================================================="

# ===========================================================================
# 11. GPU filler (optional, disabled by default for direct eval)
# ===========================================================================
GPU_FILLER="${GPU_FILLER:-0}"
FILLER_PID=""

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
# AoTBench_ReverseFilm_adaptive
#     AoTBench_UCF101_adaptive
#     AoTBench_Rtime_t2v_adaptive
#     AoTBench_Rtime_v2t_adaptive
#     AoTBench_QA_adaptive
#     MVBench_MP4_adaptive
#     Video_Holmes_adaptive
#     Video-MME_adaptive
#     ETBench_adaptive
#     MLVU_MCQ_adaptive
#     TimeLensBench_Charades_adaptive
#     TimeLensBench_ActivityNet_adaptive
#     TimeLensBench_QVHighlights_adaptive
