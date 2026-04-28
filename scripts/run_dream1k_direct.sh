#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DREAM1K_DIR="${DREAM1K_DIR:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DREAM-1K}"
export DATA="${DATA:-DREAM_local_adaptive}"
export WORK_DIR="${WORK_DIR:-/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_dream1k}"
export REUSE="${REUSE:-0}"

cd "${REPO_DIR}"
bash run_direct.sh
