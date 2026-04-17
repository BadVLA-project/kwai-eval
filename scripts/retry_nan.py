#!/usr/bin/env python
"""Retry NaN predictions: rebuild pkl from existing results, re-run inference
only for failed samples, then merge back and re-evaluate.

Usage:
    # Retry all NaN predictions in a work directory
    python scripts/retry_nan.py /path/to/eval_direct/ModelName/T20260417

    # Retry specific datasets only
    python scripts/retry_nan.py /path/to/workdir --data Video-MME_adaptive AoTBench_QA_adaptive

    # Dry run: just report NaN counts without doing anything
    python scripts/retry_nan.py /path/to/workdir --dry-run

    # Then re-run inference:
    RETRY_EMPTY=1 bash run_direct_2gpu.sh

What it does:
    1. Scans prediction .jsonl/.xlsx files in the work directory
    2. Identifies samples with NaN/empty predictions
    3. Rebuilds per-rank .pkl files with failed samples removed (so they
       get re-attempted by inference_video.py on the next run)
    4. Optionally re-evaluates after merging

The key insight: inference_video.py checks `if idx in res` to skip already-
completed samples.  By removing NaN entries from the pkl, those samples
will be re-attempted on the next run.
"""
import argparse
import glob
import json
import os
import os.path as osp
import pickle
import sys

import numpy as np
import pandas as pd


def find_prediction_files(root, data_filter=None):
    """Find prediction xlsx files (not score/acc/rating files)."""
    results = []
    for f in sorted(glob.glob(osp.join(root, '*.xlsx'))):
        basename = osp.basename(f)
        if any(x in basename for x in ['_score', '_acc', '_rating', '_submission']):
            continue
        if data_filter:
            # Check if any filter matches
            matched = False
            for d in data_filter:
                if d in basename:
                    matched = True
                    break
            if not matched:
                continue
        results.append(f)
    return results


def count_nan_predictions(filepath):
    """Count NaN/empty predictions in an xlsx or jsonl file."""
    ext = filepath.rsplit('.', 1)[-1]
    if ext == 'xlsx':
        df = pd.read_excel(filepath)
    elif ext == 'jsonl':
        records = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        return 0, 0, []

    if 'prediction' not in df.columns:
        return len(df), 0, []

    nan_mask = df['prediction'].isna() | (df['prediction'].astype(str).str.strip() == '') | \
               (df['prediction'].astype(str).str.strip().str.lower() == 'nan')
    nan_indices = df.loc[nan_mask, 'index'].tolist() if 'index' in df.columns else []
    return len(df), int(nan_mask.sum()), nan_indices


def rebuild_pkl_without_nans(workdir, xlsx_file, nan_indices):
    """Rebuild the per-rank pkl file, removing NaN entries so they get retried.

    The pkl files follow the naming pattern:
        {rank}_{ModelName}_{DatasetName}.pkl
    """
    basename = osp.basename(xlsx_file)
    stem = basename.rsplit('.', 1)[0]  # e.g. Qwen3-VL-4B-Instruct_Video-MME_adaptive

    # Look for existing pkl files
    pkl_pattern = osp.join(workdir, f'*_{stem}.pkl')
    pkl_files = sorted(glob.glob(pkl_pattern))

    if not pkl_files:
        # No pkl files — need to create them from xlsx
        # Read the xlsx to reconstruct the results dict
        df = pd.read_excel(xlsx_file)
        if 'index' not in df.columns or 'prediction' not in df.columns:
            print(f'  WARNING: {basename} missing index/prediction columns, skipping')
            return False

        # Build results dict, excluding NaN samples
        nan_set = set(nan_indices)
        res = {}
        for _, row in df.iterrows():
            idx = row['index']
            pred = row.get('prediction', '')
            if idx in nan_set:
                continue  # Skip NaN — will be retried
            if pd.isna(pred):
                continue
            res[idx] = str(pred)

        # Save as single-rank pkl (rank 0)
        pkl_path = osp.join(workdir, f'0_{stem}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(res, f)
        print(f'  Created {osp.basename(pkl_path)}: {len(res)} valid, '
              f'{len(nan_indices)} removed for retry')
        return True
    else:
        # Existing pkl files — remove NaN entries from them
        nan_set = set(nan_indices)
        total_removed = 0
        for pkl_path in pkl_files:
            with open(pkl_path, 'rb') as f:
                res = pickle.load(f)
            before = len(res)
            res = {k: v for k, v in res.items() if k not in nan_set}
            removed = before - len(res)
            total_removed += removed
            with open(pkl_path, 'wb') as f:
                pickle.dump(res, f)
            if removed > 0:
                print(f'  Updated {osp.basename(pkl_path)}: removed {removed} NaN entries')
        return total_removed > 0


def main():
    parser = argparse.ArgumentParser(description='Retry NaN predictions')
    parser.add_argument('workdir', help='Work directory containing prediction files')
    parser.add_argument('--data', nargs='+', default=None,
                        help='Only retry specific datasets (substring match)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just report NaN counts, do not modify files')
    parser.add_argument('--clean-scores', action='store_true',
                        help='Also delete *_score.* files for affected datasets')
    args = parser.parse_args()

    if not osp.isdir(args.workdir):
        print(f'ERROR: {args.workdir} is not a directory')
        sys.exit(1)

    xlsx_files = find_prediction_files(args.workdir, args.data)
    if not xlsx_files:
        print(f'No prediction files found in {args.workdir}')
        sys.exit(0)

    print(f'Scanning {len(xlsx_files)} prediction files in {args.workdir}')
    print('=' * 70)

    total_nan = 0
    affected_datasets = []

    for xlsx_path in xlsx_files:
        basename = osp.basename(xlsx_path)
        total, nan_count, nan_indices = count_nan_predictions(xlsx_path)

        if nan_count > 0:
            pct = nan_count / total * 100 if total > 0 else 0
            print(f'  {basename}: {nan_count}/{total} NaN ({pct:.1f}%)')
            total_nan += nan_count
            affected_datasets.append((xlsx_path, nan_count, nan_indices))
        else:
            print(f'  {basename}: OK ({total} samples, no NaN)')

    print('=' * 70)
    print(f'Total NaN predictions: {total_nan} across {len(affected_datasets)} datasets')

    if args.dry_run or total_nan == 0:
        return

    print()
    print('Rebuilding pkl files for retry...')
    for xlsx_path, nan_count, nan_indices in affected_datasets:
        basename = osp.basename(xlsx_path)
        print(f'\n  Processing {basename} ({nan_count} NaN samples):')
        rebuild_pkl_without_nans(args.workdir, xlsx_path, nan_indices)

        if args.clean_scores:
            stem = basename.rsplit('.', 1)[0]
            for pattern in [f'{stem}_score.*', f'{stem}_acc.*', f'{stem}_rating.*']:
                for score_file in glob.glob(osp.join(args.workdir, pattern)):
                    os.remove(score_file)
                    print(f'    Deleted {osp.basename(score_file)}')

    print()
    print('Done! To retry NaN samples, re-run your evaluation script:')
    print(f'  RETRY_EMPTY=1 bash run_direct_2gpu.sh')
    print()
    print('The inference code will:')
    print('  1. Load the rebuilt pkl files')
    print('  2. See that NaN samples are missing from the pkl')
    print('  3. Re-run inference only for those samples')
    print('  4. Merge results and re-evaluate')


if __name__ == '__main__':
    main()
