#!/bin/bash
# ==========================================================================
# run_boxed_4gpu.sh — Free boxed CoT, GPU 4-7
#
#   USE_COT=boxed: model self-decides to think, final answer in \boxed{}
#                  temperature=0.7, max_new_tokens=2048
#   GPU:           4,5,6,7  (后四卡)
#
# Usage (MUST run inside tmux/screen to survive SSH disconnection):
#   tmux new -s eval
#   bash run_boxed_4gpu.sh 2>&1 | tee ${WORK_DIR:-/tmp}/run_boxed_4gpu.log
#   # Ctrl+B D to detach, tmux attach -t eval to reattach
#
#   REUSE=1 bash run_boxed_4gpu.sh          # skip already-done samples
#   MODEL=Qwen3-VL-7B bash run_boxed_4gpu.sh
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

# ---------------------------------------------------------------------------
# Free boxed mode: USE_COT=boxed → model self-decides to think,
# requires final answer in \boxed{A}; max_new_tokens=2048
# TEMPERATURE=0 overrides the default 0.7 — no dead-loop risk per testing
# ---------------------------------------------------------------------------
export USE_COT=boxed
export TEMPERATURE=0

# ---------------------------------------------------------------------------
# GPU assignment: ranks 0-3 → physical GPUs 4-7
# ---------------------------------------------------------------------------
NGPU=4
GPU_OFFSET=4

# ---------------------------------------------------------------------------
# vLLM settings for 80GB GPUs
# ---------------------------------------------------------------------------
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-64}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

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
  Video_Holmes_64frame
  AoTBench_ReverseFilm_16frame
  AoTBench_UCF101_16frame
  AoTBench_Rtime_t2v_16frame
  AoTBench_Rtime_v2t_16frame
  AoTBench_QA_16frame
  FutureOmni_64frame
  CharadesTimeLens_1fps
  MVBench_MP4_1fps
  PerceptionTest_val_16frame
  Video-MME_64frame
  ETBench_1fps
)

MODEL="${MODEL:-Qwen3-VL-4B-Instruct}"

WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_boxed}"

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
  --model "${MODEL}"
  --work-dir "${WORK_DIR}"
)

if [ "${REUSE}" = "1" ]; then
  CMD+=(--reuse)
fi

echo "=================================================================="
echo " [$(date '+%Y-%m-%d %H:%M:%S')] run_boxed_4gpu: USE_COT=boxed  TEMPERATURE=0"
echo "   GPUs: ${GPU_OFFSET} - $((GPU_OFFSET + NGPU - 1))  (物理卡 4-7)"
echo "   work_dir: ${WORK_DIR}"
echo "=================================================================="

"${CMD[@]}"
