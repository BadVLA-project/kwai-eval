
#!/bin/bash
set -euo pipefail
set -x

# Disable torch compile to avoid startup overhead.
export TORCH_COMPILE_DISABLE=1

# Resolve libcuda lookup issues on some nodes.
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Force system toolchain for extension builds.
unset CC
unset CXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Keep tokenizer threads from oversubscribing CPU when using torchrun.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export SKIP_ERR="${SKIP_ERR:-1}"

# vLLM settings.
# VLLM_WORKER_MULTIPROC_METHOD is also set in model.py but explicit here for clarity.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Set visible GPUs; defaults to all available. Override via: CUDA_VISIBLE_DEVICES=0,1 bash run.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# GPU memory fraction given to vLLM (0.0-1.0).
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
# Tensor-parallel size. For small models (4B) TP=1 is fastest (no all-reduce overhead).
# The model.py will auto-pick based on model size if this is unset.
# For true multi-GPU speedup on 4B models, run multiple instances instead:
#   CUDA_VISIBLE_DEVICES=0 bash run.sh &
#   CUDA_VISIBLE_DEVICES=1 bash run.sh &
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"

WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/eval_2gpu}"
REUSE="${REUSE:-0}"

# Stable AoT path for Qwen3-VL-4B:
# 1. use VLMEvalKit default model config
# 2. launch with python, which is the documented path for vLLM backend
# 3. keep SKIP_ERR enabled so broken video samples do not abort the whole run
CMD=(
  python
  run.py
  --data
  AoTBench_ReverseFilm_16frame
  AoTBench_UCF101_16frame
  AoTBench_Rtime_t2v_16frame
  AoTBench_Rtime_v2t_16frame
  AoTBench_QA_16frame
  --model
  Qwen3-VL-4B-Instruct
  --work-dir "${WORK_DIR}"
  --use-vllm
)

if [ "${REUSE}" = "1" ]; then
  CMD+=(--reuse)
fi

"${CMD[@]}"
