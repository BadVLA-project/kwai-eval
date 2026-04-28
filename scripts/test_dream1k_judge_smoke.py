#!/usr/bin/env python3
"""Run a small DREAM-1K GPT judge smoke test from an existing prediction file.

This script intentionally avoids building the full DREAM dataset.  It reads a
prediction file that already contains `answer`, `events`, and `prediction`,
selects a few rows, and calls the normal `DREAM.evaluate()` path so judge
parsing, scoring, and `*_judge_io.jsonl` logging are exercised.
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ('index', 'video', 'answer', 'events', 'prediction')


def get_file_extension(file_path):
    return str(file_path).split('.')[-1]


def get_eval_output_format():
    eval_format = os.getenv('EVAL_FORMAT', '').lower()
    return eval_format if eval_format else 'csv'


def get_pred_output_format():
    pred_format = os.getenv('PRED_FORMAT', '').lower()
    return pred_format if pred_format else 'xlsx'


def get_intermediate_file_path(eval_file, suffix, target_format=None):
    original_ext = get_file_extension(eval_file)
    if target_format is None:
        if suffix.endswith(('_tmp', '_response', '_processed')):
            target_format = 'pkl'
        elif suffix.endswith(('_rating', '_config', '_meta')):
            target_format = 'json'
        elif suffix.endswith(('_acc', '_fine', '_metrics')):
            target_format = get_eval_output_format()
        else:
            target_format = get_pred_output_format()
    return str(eval_file).replace(f'.{original_ext}', f'{suffix}.{target_format}')


def read_table(path):
    ext = get_file_extension(str(path)).lower()
    if ext == 'xlsx':
        return pd.read_excel(path)
    if ext == 'tsv':
        return pd.read_csv(path, sep='\t')
    if ext == 'csv':
        return pd.read_csv(path)
    if ext == 'json':
        return pd.read_json(path)
    if ext == 'jsonl':
        return pd.read_json(path, lines=True)
    raise ValueError(f'Unsupported prediction file extension: {ext}')


def dream_video_index(value):
    value = str(value).replace('\\', '/').split('/')[-1]
    value = re.sub(r'\.mp4$', '', value, flags=re.IGNORECASE)
    return int(value) if value.isdigit() else None


def select_rows(data, limit, start, indices):
    if indices:
        wanted = [int(x) for x in indices.split(',') if x.strip()]
        return data[data['index'].astype(int).isin(wanted)].copy()
    return data.iloc[start:start + limit].copy()


def prepare_eval_files(rows, output_dir, prefix, preserve_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows.copy()

    if not preserve_index:
        mapped = rows['video'].map(dream_video_index)
        if mapped.isna().any():
            bad = rows.loc[mapped.isna(), 'video'].tolist()
            raise ValueError(f'Cannot map DREAM video names to numeric ids: {bad[:3]}')
        rows['index'] = mapped.astype(int)

    if 'question' not in rows:
        rows['question'] = 'Describe the video in detail.'

    gt_cols = ['index', 'video', 'question', 'answer', 'events']
    pred_cols = ['index', 'prediction']
    gt_df = rows[gt_cols].copy()
    pred_df = rows[pred_cols].copy()

    pred_file = output_dir / f'{prefix}.tsv'
    pred_df.to_csv(pred_file, sep='\t', index=False)
    return gt_df, pred_file


def remove_aux_files(eval_file, judge):
    for suffix, fmt in (
        (f'_{judge}_tmp', 'pkl'),
        (f'_{judge}_score', None),
        (f'_{judge}_rating', 'json'),
        (f'_{judge}_judge_io', 'jsonl'),
    ):
        path = Path(get_intermediate_file_path(str(eval_file), suffix, fmt))
        if path.exists():
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-file', required=True, help='Existing DREAM prediction xlsx/tsv/json/jsonl.')
    parser.add_argument('--output-dir', default='/tmp/dream1k_judge_smoke')
    parser.add_argument('--prefix', default='dream1k_5pred')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--indices', default='', help='Comma-separated existing indices to test instead of start/limit.')
    parser.add_argument('--preserve-index', action='store_true',
                        help='Keep the file index as-is. Default maps index from video/NNN.mp4 for DREAM parity.')
    parser.add_argument('--judge', default='gpt-4o')
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--retry', type=int, default=3)
    parser.add_argument('--use-azure-sdk', action='store_true')
    parser.add_argument('--max-completion-tokens', type=int, default=16384)
    parser.add_argument('--azure-endpoint', default=None)
    parser.add_argument('--azure-deployment-name', default=None)
    parser.add_argument('--api-version', default=None)
    parser.add_argument('--judge-args', default='{}', help='Extra JSON kwargs passed to DREAM.evaluate().')
    parser.add_argument('--force', action='store_true', help='Delete old smoke auxiliary files before running.')
    args = parser.parse_args()

    data = read_table(args.pred_file)
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f'{args.pred_file} is missing required columns: {missing}')

    rows = select_rows(data, args.limit, args.start, args.indices)
    if rows.empty:
        raise ValueError('No rows selected for smoke test.')

    output_dir = Path(args.output_dir)
    gt_df, eval_file = prepare_eval_files(rows, output_dir, args.prefix, args.preserve_index)
    if args.force:
        remove_aux_files(eval_file, args.judge)

    from vlmeval.dataset.dream import DREAM

    dataset = DREAM.__new__(DREAM)
    dataset.data = gt_df

    judge_kwargs = {
        'model': args.judge,
        'nproc': args.nproc,
        'retry': args.retry,
        **json.loads(args.judge_args),
    }
    if args.use_azure_sdk:
        judge_kwargs['use_azure_sdk'] = True
    if args.max_completion_tokens:
        judge_kwargs['max_completion_tokens'] = args.max_completion_tokens
    if args.azure_endpoint:
        judge_kwargs['azure_endpoint'] = args.azure_endpoint
    if args.azure_deployment_name:
        judge_kwargs['azure_deployment_name'] = args.azure_deployment_name
    if args.api_version:
        judge_kwargs['api_version'] = args.api_version

    print(f'[DREAM smoke] pred source: {args.pred_file}')
    print(f'[DREAM smoke] selected rows: {len(rows)}')
    print(f'[DREAM smoke] eval file: {eval_file}')
    print(f'[DREAM smoke] judge kwargs: {json.dumps({k: v for k, v in judge_kwargs.items() if k != "key"})}')

    result = dataset.evaluate(str(eval_file), **judge_kwargs)
    judge_io = get_intermediate_file_path(str(eval_file), f'_{args.judge}_judge_io', 'jsonl')
    rating = get_intermediate_file_path(str(eval_file), f'_{args.judge}_rating', 'json')
    score = get_intermediate_file_path(str(eval_file), f'_{args.judge}_score')

    print('[DREAM smoke] result:')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'[DREAM smoke] score file: {score}')
    print(f'[DREAM smoke] rating file: {rating}')
    print(f'[DREAM smoke] judge IO log: {judge_io}')
    print('[DREAM smoke] done')


if __name__ == '__main__':
    main()
