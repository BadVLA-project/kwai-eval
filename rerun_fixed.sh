#!/bin/bash
# ==========================================================================
# rerun_fixed.sh — 修复后重跑流程
#
# 用法:
#   bash rerun_fixed.sh              # dry-run, 只打印要执行的操作
#   bash rerun_fixed.sh --execute    # 真正执行
#
# 流程:
#   1. 删除 ETBench/MLVU 的旧 TSV 缓存（因 index/candidates 格式 bug）
#   2. 删除 ETBench/MLVU 的旧推理结果（因 prompt 错误需重新推理）
#   3. 用 REUSE=1 跑全部数据集（已完成的自动复用，未完成的继续跑）
#
# 可选环境变量:
#   WORK_DIR    — 评测输出目录 (默认: /m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct)
#   MODEL       — 模型名 (默认: Qwen3-VL-4B-Instruct)
#   EVAL_ID     — 固定 eval_id (默认: 自动按日期)
#   DRY_RUN     — 1=只打印不执行 (默认: 1, 用 --execute 切换为 0)
# ==========================================================================
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
if [[ "${1:-}" == "--execute" ]]; then
  DRY_RUN=0
fi

WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct}"
MLVU_DIR="${MLVU_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/MLVU_Test}"
ETBENCH_DIR="${ETBENCH_DIR:-/m2v_intern/xuboshen/zgw/Benchmarks/ETBench}"

# Auto-detect all model directories, or override with MODELS="m1 m2"
if [[ -n "${MODELS:-}" ]]; then
  read -ra MODEL_LIST <<< "${MODELS}"
else
  MODEL_LIST=()
  if [[ -d "${WORK_DIR}" ]]; then
    for d in "${WORK_DIR}"/*/; do
      [[ -d "$d" ]] || continue
      MODEL_LIST+=("$(basename "$d")")
    done
  fi
fi

if [[ ${#MODEL_LIST[@]} -eq 0 ]]; then
  echo "  [ERROR] 没有找到模型目录。请设置 MODELS 环境变量。"
  exit 1
fi

# Datasets whose TSV + predictions are buggy (prompt was wrong)
BUGGY_DATASETS=(MLVU_MCQ ETBench)

# All prediction patterns to clean for buggy datasets
# (matches files like Model_MLVU_MCQ_adaptive.xlsx, etc.)
BUGGY_PRED_PATTERNS=(MLVU_MCQ ETBench)

echo "=================================================================="
echo " rerun_fixed.sh — 修复后重跑流程"
echo "   WORK_DIR:  ${WORK_DIR}"
echo "   MODELS:    ${MODEL_LIST[*]}"
echo "   DRY_RUN:   ${DRY_RUN} (用 --execute 实际执行)"
echo "=================================================================="

run_cmd() {
  echo "  [CMD] $*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

# -----------------------------------------------------------------------
# Step 1: 删除旧 TSV 缓存
# -----------------------------------------------------------------------
echo ""
echo "=== Step 1: 删除旧 TSV 缓存 ==="

TSV_FILES=(
  "${MLVU_DIR}/MLVU_MCQ.tsv"
  "${ETBENCH_DIR}/ETBench.tsv"
  "${ETBENCH_DIR}/ETBench_subset.tsv"
)

for tsv in "${TSV_FILES[@]}"; do
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [DELETE] ${tsv}  (如果存在)"
  else
    if [[ -f "${tsv}" ]]; then
      rm -v "${tsv}"
    else
      echo "  [SKIP] ${tsv} (不存在)"
    fi
  fi
done

# -----------------------------------------------------------------------
# Step 2: 删除 buggy 数据集的旧推理结果
# -----------------------------------------------------------------------
echo ""
echo "=== Step 2: 删除 ETBench/MLVU 的旧推理结果 ==="

for MODEL in "${MODEL_LIST[@]}"; do
  echo "  --- Model: ${MODEL} ---"
  MODEL_DIR="${WORK_DIR}/${MODEL}"
  if [[ -d "${MODEL_DIR}" ]]; then
    for eval_dir in "${MODEL_DIR}"/T*; do
      [[ -d "${eval_dir}" ]] || continue
      for pattern in "${BUGGY_PRED_PATTERNS[@]}"; do
        matches=()
        while IFS= read -r -d '' f; do
          matches+=("$f")
        done < <(find "${eval_dir}" -maxdepth 1 -name "${MODEL}_${pattern}*" -print0 2>/dev/null)

        if [[ ${#matches[@]} -gt 0 ]]; then
          for f in "${matches[@]}"; do
            if [[ "${DRY_RUN}" == "1" ]]; then
              echo "  [DELETE] ${f}"
            else
              rm -v "${f}"
            fi
          done
        fi
      done
    done
  else
    echo "  [SKIP] ${MODEL_DIR} 不存在"
  fi
done

# -----------------------------------------------------------------------
# Step 3: 用 REUSE=1 跑全部数据集
# -----------------------------------------------------------------------
echo ""
echo "=== Step 3: REUSE=1 跑全部数据集 ==="
echo "  已完成且无 bug 的数据集 → 自动复用推理结果"
echo "  ETBench/MLVU → 重新推理 (旧结果已删)"
echo "  未完成的 → 继续跑"
echo ""
echo "  注意: Step 3 只跑 run_direct_2gpu.sh 里配置的模型。"
echo "  如需跑其他模型，修改 run_direct_2gpu.sh 的 MODELS 或用 MODEL=xxx 环境变量。"
echo ""

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "  [CMD] REUSE=1 bash run_direct_2gpu.sh"
  echo ""
  echo ">>> 确认无误后，执行: bash rerun_fixed.sh --execute"
else
  echo "  启动评测..."
  REUSE=1 bash run_direct_2gpu.sh
fi
