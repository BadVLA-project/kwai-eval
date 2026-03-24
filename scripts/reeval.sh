#!/bin/bash
# Re-evaluate existing inference results without re-running inference.
#
# Two modes:
#
# 1) Legacy (flag) mode — pass run.py arguments directly:
#      bash scripts/reeval.sh --model Qwen3-VL-72B --data VideoMME AoTBench_ReverseFilm ...
#
# 2) Directory mode — auto-discover models/datasets from work-dirs:
#      bash scripts/reeval.sh dir1 dir2 dir3 dir4
#      bash scripts/reeval.sh dir1 dir2 -- --judge chatgpt-0125
#      bash scripts/reeval.sh dir1 --dry-run
#
#    Walks each directory recursively, finds all prediction xlsx files,
#    resolves dataset names from the filename, and calls cls.evaluate()
#    directly.  Works with any layout (bench-centric, model-centric,
#    symlinks, etc.).
#
# What it does (legacy mode):
#   1. --reuse        : keep existing prediction xlsx (skip inference)
#   2. --reuse-aux 0  : delete old score / rating / acc files -> force re-scoring
#   3. --mode eval    : skip inference stage entirely
#
# What it does (directory mode):
#   Uses scripts/reeval.py — directly calls dataset evaluate classmethods.
#   No inference, no prepare_dataset(), works on any directory structure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Legacy mode: first arg starts with -- ----
if [ "${1:-}" != "${1#--}" ]; then
    exec python run.py --reuse --reuse-aux 0 --mode eval "$@"
fi

# ---- Directory mode: delegate to reeval.py ----
# Split args: dirs (before --) and extra flags (after --)
dirs=()
extra=()
past_sep=false
for a in "$@"; do
    if [ "$a" = "--" ]; then
        past_sep=true
        continue
    fi
    if $past_sep; then
        extra+=("$a")
    else
        dirs+=("$a")
    fi
done

if [ ${#dirs[@]} -eq 0 ]; then
    echo "Usage:"
    echo "  bash scripts/reeval.sh dir1 [dir2 ...] [-- --judge chatgpt-0125]"
    echo "  bash scripts/reeval.sh --model MODEL --data DS1 DS2 ..."
    exit 1
fi

if [ ${#extra[@]} -gt 0 ]; then
    exec python "$SCRIPT_DIR/reeval.py" "${dirs[@]}" "${extra[@]}"
else
    exec python "$SCRIPT_DIR/reeval.py" "${dirs[@]}"
fi
