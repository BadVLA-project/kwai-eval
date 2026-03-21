#!/bin/bash
set -euo pipefail
set -x

# ==========================================================================
# run_vllm.sh — vLLM backend, runs both CoT and No-CoT in one script.
#
# Usage:
#   bash run_vllm.sh                  # run both CoT + No-CoT
#   USE_COT=1 bash run_vllm.sh       # CoT only
#   USE_COT=0 bash run_vllm.sh       # No-CoT only
# ==========================================================================

# Disable torch compile to avoid startup overhead.
export TORCH_COMPILE_DISABLE=1

# Force all distributed / vLLM networking to use loopback (single-node).
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export VLLM_HOST_IP=127.0.0.1

# Resolve libcuda lookup issues on some nodes.
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Force system toolchain for extension builds.
unset CC
unset CXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Keep tokenizer threads from oversubscribing CPU.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SKIP_ERR="${SKIP_ERR:-1}"

REUSE="${REUSE:-0}"

# ---------------------------------------------------------------------------
# Offline mode: use local HF cache, disable all network access.
# ---------------------------------------------------------------------------
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=0
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# ---------------------------------------------------------------------------
# Local dataset overrides (skip HuggingFace download).
# ---------------------------------------------------------------------------
export VIDEO_HOLMES_DIR="${VIDEO_HOLMES_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/Video-Holmes}"
export TIMELENS_DIR="${TIMELENS_DIR:-/m2v_intern/xuboshen/zgw/hf_cache_temp/TimeLens-Bench}"
export PERCEPTION_TEST_DIR="${PERCEPTION_TEST_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/PerceptionTest}"
export MLVU_DIR="${MLVU_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/MLVU_Test}"

# ---------------------------------------------------------------------------
# Multi-GPU data-parallel launch via torchrun.
# ---------------------------------------------------------------------------
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

# ---------------------------------------------------------------------------
# vLLM settings for 80GB GPUs (Qwen3-VL-4B ~10GB VRAM, tp=1).
# ---------------------------------------------------------------------------
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-64}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

# ---------------------------------------------------------------------------
# Datasets & Models
# ---------------------------------------------------------------------------
DATASETS=(
  AoTBench_ReverseFilm_16frame
  AoTBench_UCF101_16frame
  AoTBench_Rtime_t2v_16frame
  AoTBench_Rtime_v2t_16frame
  AoTBench_QA_16frame
  FutureOmni_64frame
  CharadesTimeLens_1fps
  MVBench_MP4_1fps
  PerceptionTest_val_16frame
)

MODELS=(
  Qwen3-VL-4B-Instruct
  Qwen3-VL-4B-Instruct_aot_ablation_exp1_v2t_binary
  Qwen3-VL-4B-Instruct_aot_ablation_exp2_v2t_3way
  Qwen3-VL-4B-Instruct_aot_ablation_exp3_t2v_binary
  Qwen3-VL-4B-Instruct_aot_ablation_exp4_t2v_3way
)

# ---------------------------------------------------------------------------
# Helper: launch one evaluation run.
# ---------------------------------------------------------------------------
run_eval() {
  local cot_flag="$1"
  local work_dir="$2"

  export USE_COT="${cot_flag}"

  local CMD=(
    torchrun
    --nproc-per-node="${NGPU}"
    --master-addr="${MASTER_ADDR:-127.0.0.1}"
    --master-port="${MASTER_PORT:-29500}"
    run.py
    --use-vllm
    --data "${DATASETS[@]}"
    --model "${MODELS[@]}"
    --work-dir "${work_dir}"
  )

  if [ "${REUSE}" = "1" ]; then
    CMD+=(--reuse)
  fi

  "${CMD[@]}"
}

# ---------------------------------------------------------------------------
# Main: run CoT, No-CoT, or both depending on USE_COT env var.
# ---------------------------------------------------------------------------
WORK_DIR_BASE="${WORK_DIR_BASE:-/m2v_intern/xuboshen/zgw/VideoProxyMixed}"

if [ -z "${USE_COT:-}" ]; then
  # No USE_COT specified — run both
  echo "=== Phase 1/2: No-CoT (greedy, temperature=0) ==="
  run_eval 0 "${WORK_DIR_BASE}/evaluation_vllm_noCoT"

  echo "=== Phase 2/2: CoT (sampling, temperature=0.7) ==="
  run_eval 1 "${WORK_DIR_BASE}/evaluation_vllm_CoT"
elif [ "${USE_COT}" = "0" ]; then
  echo "=== No-CoT only (greedy, temperature=0) ==="
  run_eval 0 "${WORK_DIR_BASE}/evaluation_vllm_noCoT"
elif [ "${USE_COT}" = "1" ]; then
  echo "=== CoT only (sampling, temperature=0.7) ==="
  run_eval 1 "${WORK_DIR_BASE}/evaluation_vllm_CoT"
else
  echo "ERROR: USE_COT must be 0, 1, or unset (for both). Got: ${USE_COT}" >&2
  exit 1
fi
