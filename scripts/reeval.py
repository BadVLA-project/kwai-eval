#!/usr/bin/env python
"""Direct re-evaluation of prediction files across arbitrary directory layouts.

Usage:
    python scripts/reeval.py dir1 dir2 dir3 dir4
    python scripts/reeval.py dir1 dir2 --judge chatgpt-0125
    python scripts/reeval.py dir1 --dry-run           # list what would be evaluated
    python scripts/reeval.py dir1 --force-clean=false  # keep existing score files

Works with any directory structure (bench-centric, model-centric, or mixed with
symlinks).  Resolves symlinks to avoid duplicate evaluation.  Calls
cls.evaluate(file) directly — no inference, no prepare_dataset().
"""
import argparse
import glob
import os
import os.path as osp
import sys
import traceback

# Project root = parent of scripts/
_PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


def _build_dataset_registry():
    """Return {dataset_name: class} for all known datasets."""
    from vlmeval.dataset.video_dataset_config import supported_video_datasets
    from vlmeval.dataset import DATASET_CLASSES

    registry = {}

    # Path B: video config (partial objects) — these include _Nframe suffixed names
    for name, partial_fn in supported_video_datasets.items():
        registry[name] = partial_fn.func

    # Path A: DATASET_CLASSES (base names like AoTBench_ReverseFilm, Video-MME)
    for cls in DATASET_CLASSES:
        for name in cls.supported_datasets():
            if name not in registry:
                registry[name] = cls

    return registry


def find_prediction_files(root):
    """Find all prediction xlsx files under *root*, excluding score/acc/submission."""
    results = []
    for f in glob.glob(osp.join(root, '**', '*.xlsx'), recursive=True):
        base = osp.splitext(osp.basename(f))[0]
        if base.endswith(('_score', '_acc', '_submission')):
            continue
        real = osp.realpath(f)
        if osp.isfile(real):
            results.append(real)
    # De-duplicate (same real file reached via symlinks)
    return sorted(set(results))


def infer_dataset_name(filepath, known_names):
    """Extract dataset name from a filename like Model_Dataset.xlsx.

    Tries longest known name first so 'Video-MME_64frame_short_medium' matches
    before 'Video-MME'.
    """
    stem = osp.splitext(osp.basename(filepath))[0]
    for ds in sorted(known_names, key=len, reverse=True):
        suffix = '_' + ds
        if stem.endswith(suffix):
            model = stem[:-len(suffix)]
            if model:  # model part must be non-empty
                return model, ds
    return None, None


def clean_score_files(pred_file):
    """Remove existing score/acc/rating files to force re-evaluation."""
    base = pred_file.rsplit('.', 1)[0]
    removed = []
    for suffix in ['_score', '_acc', '_rating']:
        for ext in ['xlsx', 'jsonl', 'json', 'csv', 'pkl']:
            pattern = f'{base}{suffix}.{ext}'
            for f in glob.glob(pattern):
                os.remove(f)
                removed.append(osp.basename(f))
    return removed


def main():
    parser = argparse.ArgumentParser(
        description='Re-evaluate prediction files across directories.')
    parser.add_argument('dirs', nargs='+', help='Directories to scan')
    parser.add_argument('--judge', default=None,
                        help='Judge model (e.g. chatgpt-0125). Default: exact_matching')
    parser.add_argument('--dry-run', action='store_true',
                        help='List discoveries without running evaluation')
    parser.add_argument('--force-clean', default='true', choices=['true', 'false'],
                        help='Delete existing score files before re-eval (default: true)')
    args = parser.parse_args()
    do_clean = args.force_clean == 'true'

    registry = _build_dataset_registry()
    known_names = set(registry.keys())

    total, ok, skip, fail = 0, 0, 0, 0

    for d in args.dirs:
        d = osp.abspath(d)
        print(f'\n{"=" * 60}')
        print(f'  Scanning: {d}')
        print(f'{"=" * 60}')

        if not osp.isdir(d):
            print('  WARNING: not a directory, skipping')
            skip += 1
            continue

        files = find_prediction_files(d)
        print(f'  Found {len(files)} prediction file(s) (after dedup)')

        for f in files:
            model, ds = infer_dataset_name(f, known_names)
            if ds is None:
                # Try without frame suffix: e.g. some files may be manually renamed
                continue
            total += 1
            cls = registry.get(ds)

            print(f'\n  [{total}] {osp.basename(f)}')
            print(f'       model={model}  dataset={ds}')
            print(f'       class={cls.__name__ if cls else "N/A"}')

            if cls is None:
                print('       SKIP: no evaluate class found')
                skip += 1
                continue

            if args.dry_run:
                print('       (dry run)')
                continue

            if do_clean:
                removed = clean_score_files(f)
                if removed:
                    print(f'       cleaned: {", ".join(removed)}')

            judge_kwargs = {}
            if args.judge:
                judge_kwargs['model'] = args.judge

            try:
                cls.evaluate(f, **judge_kwargs)
                print('       OK')
                ok += 1
            except Exception as e:
                print(f'       FAIL: {e}')
                traceback.print_exc()
                fail += 1

    print(f'\n{"=" * 60}')
    print(f'  Summary: total={total}  ok={ok}  skip={skip}  fail={fail}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
