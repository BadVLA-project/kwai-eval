#!/bin/bash
set -euo pipefail
set -x

# ==========================================================================
# run_cache_test.sh — Test frame cache on VideoMME short+medium
#
# Runs Qwen3-VL-4B on VideoMME (short+medium only) twice:
#   1st run: cold cache (frames decoded and saved)
#   2nd run: warm cache (frames loaded from disk)
#
# Usage:
#   bash run_cache_test.sh
# ==========================================================================

export TORCH_COMPILE_DISABLE=1
export VLLM_HOST_IP=127.0.0.1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
unset CC CXX
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SKIP_ERR="${SKIP_ERR:-1}"

# HF cache
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=0
export HF_HOME="${HF_HOME:-/m2v_intern/xuboshen/zgw/hf_cache_temp}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# vLLM settings
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_BATCH_CHUNK_SIZE="${VLLM_BATCH_CHUNK_SIZE:-64}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
DELAY="${DELAY:-15}"
WORK_DIR_BASE="${WORK_DIR_BASE:-/m2v_intern/xuboshen/zgw/VideoProxyMixed}"

CONFIG_FILE="config_cache_test.json"

# ---------------------------------------------------------------------------
# Generate config JSON: VideoMME short+medium only, with frame cache ON
# ---------------------------------------------------------------------------
cat > "${CONFIG_FILE}" <<'JSONEOF'
{
    "model": {
        "Qwen3-VL-4B-Instruct": {}
    },
    "data": {
        "Video-MME_64frame_short_medium": {
            "class": "VideoMME",
            "dataset": "Video-MME",
            "nframe": 64,
            "durations": ["short", "medium"]
        }
    }
}
JSONEOF

echo "=========================================="
echo "  Run 1: Cold cache (first-time decode)"
echo "=========================================="

WORK_DIR_RUN1="${WORK_DIR_BASE}/cache_test_run1"

python launch_workers.py \
    --ngpu "${NGPU}" \
    --delay "${DELAY}" \
    -- \
    run.py \
    --use-vllm \
    --config "${CONFIG_FILE}" \
    --work-dir "${WORK_DIR_RUN1}" \
    --mode infer

echo ""
echo "=========================================="
echo "  Run 2: Warm cache (cached frames)"
echo "=========================================="

WORK_DIR_RUN2="${WORK_DIR_BASE}/cache_test_run2"

python launch_workers.py \
    --ngpu "${NGPU}" \
    --delay "${DELAY}" \
    -- \
    run.py \
    --use-vllm \
    --config "${CONFIG_FILE}" \
    --work-dir "${WORK_DIR_RUN2}" \
    --mode infer

echo ""
echo "=========================================="
echo "  Timing comparison"
echo "=========================================="
echo "Run 1 (cold cache):"
cat "${WORK_DIR_RUN1}/timing.log" 2>/dev/null || echo "  (no timing.log found)"
echo ""
echo "Run 2 (warm cache):"
cat "${WORK_DIR_RUN2}/timing.log" 2>/dev/null || echo "  (no timing.log found)"

# Cleanup config
rm -f "${CONFIG_FILE}"

echo ""
echo "Done. Compare timing above to see cache speedup."
echo "Cache dir: check \$(LMUDataRoot)/model_frame_cache/"
