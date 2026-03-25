#!/usr/bin/env python3
"""Generate all visualization figures for video benchmark evaluation.

Usage:
    python -m vis.main --work_dir /path/to/WORK_DIR [--output_dir vis/output] [--format png]
"""

import argparse
import os
import sys

from .style import apply_style
from .data_loader import ResultLoader
from .plot_heatmap import plot_master_heatmap, plot_aot_heatmap, plot_videomme_duration_heatmap
from .plot_radar import plot_aot_radar, plot_tg_radar
from .plot_bar import plot_overall_bar, plot_delta_bar, plot_aot_focused_bar
from .plot_breakdown import (
    plot_mvbench_breakdown, plot_videomme_tasktype,
    plot_videoholmes_breakdown, plot_perceptiontest_breakdown,
    plot_charades_breakdown,
)


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation visualizations')
    parser.add_argument('--work_dir', required=True, help='Path to WORK_DIR with score files')
    parser.add_argument('--output_dir', default='vis/output', help='Output directory for figures')
    parser.add_argument('--format', nargs='+', default=['png'], help='Output figure formats')
    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        print(f'ERROR: work_dir does not exist: {args.work_dir}')
        sys.exit(1)

    apply_style()
    loader = ResultLoader(args.work_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    fmts = tuple(args.format)

    print(f'Work dir:   {args.work_dir}')
    print(f'Output dir: {args.output_dir}')
    print(f'Formats:    {fmts}')
    print()

    # ── Heatmaps ─────────────────────────────────────────────────────────
    print('[1/13] Master heatmap (models x benchmarks)')
    plot_master_heatmap(loader, args.output_dir, fmts)

    print('[2/13] AoT heatmap (models x 5 AoT variants)')
    plot_aot_heatmap(loader, args.output_dir, fmts)

    print('[3/13] VideoMME duration heatmap')
    plot_videomme_duration_heatmap(loader, args.output_dir, fmts)

    # ── Radar charts ─────────────────────────────────────────────────────
    print('[4/13] AoT ablation radar')
    plot_aot_radar(loader, args.output_dir, fmts)

    print('[5/13] TG ablation radar')
    plot_tg_radar(loader, args.output_dir, fmts)

    # ── Bar charts ───────────────────────────────────────────────────────
    print('[6/13] Overall bar chart')
    plot_overall_bar(loader, args.output_dir, fmts)

    print('[7/13] Delta bar chart (vs base)')
    plot_delta_bar(loader, args.output_dir, fmts)

    print('[8/13] AoT focused bar with error bars')
    plot_aot_focused_bar(loader, args.output_dir, fmts)

    # ── Breakdowns ───────────────────────────────────────────────────────
    print('[9/13] MVBench 20 sub-tasks breakdown')
    plot_mvbench_breakdown(loader, args.output_dir, fmts)

    print('[10/13] VideoMME task type breakdown')
    plot_videomme_tasktype(loader, args.output_dir, fmts)

    print('[11/13] Video-Holmes question type breakdown')
    plot_videoholmes_breakdown(loader, args.output_dir, fmts)

    print('[12/13] PerceptionTest multi-dimension breakdown')
    plot_perceptiontest_breakdown(loader, args.output_dir, fmts)

    print('[13/13] CharadesTimeLens metrics breakdown')
    plot_charades_breakdown(loader, args.output_dir, fmts)

    print()
    print(f'Done! All figures saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
