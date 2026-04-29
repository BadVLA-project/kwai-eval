#!/usr/bin/env python3
"""Eval dashboard — auto-discovers models and benchmarks from work_dir.

Usage:
    python -m vis.main --work-dir /m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct              # serve at port 8890
    python -m vis.main --work-dir /path/to/results --port 9000  # custom port
    python -m vis.main --work-dir /path/to/results -o out.html  # export HTML file
"""

import argparse

from .data_loader import ResultLoader
from .web import generate_dashboard, serve


def main():
    parser = argparse.ArgumentParser(description='Eval visualization dashboard')
    parser.add_argument('--work-dir', '--work_dir', required=True, help='Results directory')
    parser.add_argument('-o', '--output', default=None, help='Export HTML file instead of serving')
    parser.add_argument('--port', type=int, default=8890, help='HTTP server port (default: 8890)')
    parser.add_argument('--include-overlap', action='store_true', help='Include slow row-level overlap analysis')
    args = parser.parse_args()

    loader = ResultLoader(args.work_dir)
    print(f'Discovered {len(loader.models)} models, {len(loader.benchmarks)} benchmarks')
    if loader.models:
        print(f'  Models: {", ".join(loader.models[:5])}{"..." if len(loader.models) > 5 else ""}')
    if loader.benchmarks:
        print(f'  Benchmarks: {", ".join(loader.benchmarks[:5])}{"..." if len(loader.benchmarks) > 5 else ""}')

    if args.output:
        generate_dashboard(loader, args.output, include_overlap=args.include_overlap)
    else:
        serve(loader, args.port, include_overlap=args.include_overlap)


if __name__ == '__main__':
    main()
