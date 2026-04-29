#!/usr/bin/env python3
"""Build a multi-model benchmark answer-overlap report from row-level scores."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vis.overlap_analysis import build_overlap_analysis, write_overlap_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze answer overlap across evaluated models")
    parser.add_argument("--work-dir", required=True, help="Evaluation result root directory")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model directory names")
    parser.add_argument("--baseline", default=None, help="Baseline model for delta/subgroup analysis")
    parser.add_argument("--data", nargs="*", default=None, help="Optional benchmark names")
    parser.add_argument("--group-columns", nargs="*", default=None, help="Explicit subgroup columns")
    parser.add_argument("--min-group-size", type=int, default=5, help="Minimum subgroup size for reports")
    parser.add_argument(
        "--all-baselines",
        action="store_true",
        help="Compute dataset/subgroup deltas for every ordered model pair",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.5,
        help="Threshold for treating continuous score/iou metrics as correct",
    )
    parser.add_argument(
        "--max-case-matrix",
        type=int,
        default=None,
        help="Optional cap for full case_matrix.jsonl rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = build_overlap_analysis(
        work_dir=args.work_dir,
        models=args.models,
        baseline=args.baseline,
        data=args.data,
        group_columns=args.group_columns,
        min_group_size=args.min_group_size,
        correct_threshold=args.correct_threshold,
        max_case_matrix=args.max_case_matrix,
        all_baselines=args.all_baselines,
    )
    artifacts = write_overlap_bundle(analysis, args.out_dir)
    print(f"Models: {len(analysis['models'])}")
    print(f"Datasets: {len(analysis['datasets'])}")
    print(f"Pairwise rows: {len(analysis['pairwise_overlap'])}")
    print(f"Group delta rows: {len(analysis['group_deltas'])}")
    print(f"Report written to: {os.path.abspath(args.out_dir)}")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
