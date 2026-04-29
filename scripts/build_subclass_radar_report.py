#!/usr/bin/env python3
"""Build per-benchmark subclass radar charts from aggregate eval results."""

from __future__ import annotations

import argparse
import os.path as osp
import sys

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vis.subclass_radar import DEFAULT_BENCHMARKS, build_subclass_radar_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subclass radar report for selected benchmarks")
    parser.add_argument("--work-dir", required=True, help="Evaluation result root directory")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--benchmarks", nargs="*", default=DEFAULT_BENCHMARKS, help="Benchmark families")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model directory names")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_subclass_radar_report(
        work_dir=args.work_dir,
        out_dir=args.out_dir,
        benchmarks=args.benchmarks,
        models=args.models,
    )
    print(f"Report written to: {osp.abspath(args.out_dir)}")
    for bench in report["benchmarks"]:
        info = report["by_bench"][bench]
        if info.get("skipped"):
            print(f"- {bench}: skipped ({info.get('reason', 'no subclass breakdown')})")
        else:
            print(f"- {bench}: {len(info['dimensions'])} subclasses -> by_bench/{bench}/radar.png")


if __name__ == "__main__":
    main()
