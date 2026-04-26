#!/usr/bin/env python
"""Patch fast ETBench threshold metrics into existing result files.

This script does not run model inference and does not recompute slow text
caption metrics such as BLEU, METEOR, CIDEr, or SentSim. It reads existing
ETBench prediction files, recomputes only time-overlap metrics for GND and
CAP(F1), then updates the sibling ``*_score.json`` and ``*_etbench_acc.csv``.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import os.path as osp
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd

IOU_THRESHOLDS = ('0.1', '0.3', '0.5', '0.7')
GND_TASKS = ('tvg', 'epm', 'tal', 'evs', 'vhd')
CAP_TASKS = ('dvc', 'slc')
PRED_EXTS = ('.xlsx', '.jsonl', '.json', '.csv', '.tsv')
EXCLUDED_STEM_SUFFIXES = (
    '_score',
    '_acc',
    '_rating',
    '_submission',
    '_etbench_acc',
)


def _parse_spans(text: str) -> list[list[float]]:
    spans = []
    for pat in (
        r'(\d+\.?\d*)\s*[-\u2013\u2014]\s*(\d+\.?\d*)\s*(?:seconds?)?',
        r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',
    ):
        for match in re.finditer(pat, text, re.IGNORECASE):
            start, end = float(match.group(1)), float(match.group(2))
            spans.append([min(start, end), max(start, end)])

    seen = set()
    unique = []
    for span in spans:
        key = (span[0], span[1])
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def _np_temporal_iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    inter_s = np.maximum(gt[:, 0:1], pred[:, 0])
    inter_e = np.minimum(gt[:, 1:2], pred[:, 1])
    inter = np.maximum(0.0, inter_e - inter_s)
    gt_len = gt[:, 1] - gt[:, 0]
    pred_len = pred[:, 1] - pred[:, 0]
    union = gt_len[:, None] + pred_len[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _iou_interval(a: list[float], b: list[float]) -> float:
    start_i, end_i = a[0], a[1]
    start, end = b[0], b[1]
    intersection = max(0.0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), (end - start) + (end_i - start_i))
    return float(intersection) / (union + 1e-8)


def _tvg_eval(samples: list[dict[str, Any]]) -> dict[str, float]:
    hit = [0] * len(IOU_THRESHOLDS)
    failed, sum_iou = 0, 0.0
    for sample in samples:
        pred_spans = _parse_spans(sample['pred'])
        if not pred_spans:
            failed += 1
            continue
        gt = np.array(sample['tgt'], dtype=float)
        pred = np.array([pred_spans[0]], dtype=float)
        iou = float(_np_temporal_iou(gt, pred).max())
        sum_iou += iou
        for idx, thr in enumerate(IOU_THRESHOLDS):
            if iou >= float(thr):
                hit[idx] += 1

    recall = [h / len(samples) for h in hit] if samples else [0.0] * len(IOU_THRESHOLDS)
    out: dict[str, float] = {
        'Total': len(samples),
        'Failed': failed,
        'mIoU': round(sum_iou / len(samples), 5) if samples else 0.0,
    }
    for rec, thr in zip(recall, IOU_THRESHOLDS):
        out[f'F1@{thr}'] = round(rec, 5)
    out['F1'] = round(sum(recall) / len(recall), 5) if recall else 0.0
    return out


def _tal_eval(samples: list[dict[str, Any]]) -> dict[str, float]:
    f1_scores = [0.0] * len(IOU_THRESHOLDS)
    failed = 0
    for sample in samples:
        pred_spans = _parse_spans(sample['pred'])
        if not pred_spans:
            failed += 1
            continue
        gt = np.array(sample['tgt'], dtype=float)
        pred = np.array(pred_spans, dtype=float)
        iou = _np_temporal_iou(gt, pred)
        for idx, thr in enumerate(IOU_THRESHOLDS):
            thr_f = float(thr)
            if iou.max() < thr_f:
                continue
            rec = float((iou.max(axis=1) >= thr_f).mean())
            prc = float((iou.max(axis=0) >= thr_f).mean())
            f1_scores[idx] += 2 * prc * rec / (prc + rec)

    if samples:
        f1_scores = [score / len(samples) for score in f1_scores]
    out: dict[str, float] = {'Total': len(samples), 'Failed': failed}
    for f1, thr in zip(f1_scores, IOU_THRESHOLDS):
        out[f'F1@{thr}'] = round(f1, 5)
    out['F1'] = round(sum(f1_scores) / len(f1_scores), 5) if f1_scores else 0.0
    return out


def _evs_eval(samples: list[dict[str, Any]]) -> dict[str, float]:
    f1_scores = []
    failed = 0
    for sample in samples:
        pred_spans = _parse_spans(sample['pred'])
        if not pred_spans:
            failed += 1
            continue
        gt_map = np.zeros(1000)
        for gt in sample['tgt']:
            gt_map[max(0, round(gt[0])):round(gt[1])] = 1
        pred_map = np.zeros(1000)
        for pred in pred_spans:
            pred_map[max(0, round(pred[0])):round(pred[1])] = 2
        merged = gt_map + pred_map
        tp = float((merged == 3).sum())
        fp = float((merged == 2).sum())
        fn = float((merged == 1).sum())
        if tp == 0:
            f1_scores.append(0.0)
        else:
            rec = tp / (tp + fn)
            prc = tp / (tp + fp)
            f1_scores.append(2 * prc * rec / (prc + rec))
    return {
        'Total': len(samples),
        'Failed': failed,
        'F1': round(sum(f1_scores) / len(f1_scores), 5) if f1_scores else 0.0,
    }


def _vhd_eval(samples: list[dict[str, Any]]) -> dict[str, float]:
    hit, failed = 0, 0
    for sample in samples:
        gt = sample['tgt']
        if gt and not isinstance(gt[0][0], (list, tuple)):
            gt = [gt]
        match = re.search(r'[-+]?\d*\.?\d+', sample['pred'])
        if not match:
            failed += 1
            continue
        pred = float(match.group(0))
        if any(start <= pred <= end for annotator in gt for start, end in annotator):
            hit += 1
    return {
        'Total': len(samples),
        'Failed': failed,
        'F1': round(hit / len(samples), 5) if samples else 0.0,
    }


def _extract_time_part(time_part: str) -> list[str]:
    radius = 20
    result = re.compile(r'\d+\.*\d*\s*-\s*\d+\.*\d*').findall(time_part)
    if not result:
        if time_part.count(':') == 1:
            times = re.compile(r'\d+\.*\d*:\d+\.*\d*').findall(time_part)
            if times:
                t = int(times[0].split(':')[0]) * 60 + int(times[0].split(':')[1])
                result = [f'{max(0, t - radius)} - {t + radius}']
        elif time_part.count(':') == 2:
            times = re.compile(r'\d+\.*\d*:\d+\.*\d*').findall(time_part)
            if len(times) == 2:
                start = int(times[0].split(':')[0]) * 60 + int(times[0].split(':')[1])
                end = int(times[1].split(':')[0]) * 60 + int(times[1].split(':')[1])
                result = [f'{start} - {end}']
    if not result:
        nums = re.compile(r'\d+\.*\d*(?!\.)').findall(time_part)
        if len(nums) == 1:
            t = float(nums[0])
            result = [f'{max(0, t - radius)} - {t + radius}']
        elif len(nums) == 2:
            result = [f'{nums[0]} - {nums[1]}']
    return result


def _extract_time_from_paragraph(paragraph: str) -> tuple[list[list[float]], list[str]]:
    paragraph = paragraph.lower()
    timestamps, captions = [], []
    time_matches = re.findall(r'(\d+\.*\d*)\s*-\s*(\d+\.*\d*)', paragraph)
    string_matches = re.findall(r'(\d+\.*\d*\s*-\s*\d+\.*\d*)', paragraph)
    if time_matches:
        timestamps = [[float(start), float(end)] for start, end in time_matches]
        rest = paragraph
        for time_string in string_matches:
            rest = rest.replace(time_string, '\n')
        captions = rest.replace('seconds', '').split('\n')

    if not timestamps:
        start_pat = r'(?:start(?:ing)?\s+time:\s*(\d+\.*\d*)(?:s|\s+seconds)?)'
        end_pat = r'(?:end(?:ing)?\s+time:\s*(\d+\.*\d*)(?:s|\s+seconds)?)'
        starts = re.findall(start_pat, paragraph, re.IGNORECASE)
        ends = re.findall(end_pat, paragraph, re.IGNORECASE)
        if starts and ends:
            timestamps = [[float(start), float(end)] for start, end in zip(starts, ends)]
            captions = re.findall(r'description:\s*(.*)', paragraph) or re.findall(r'\*\s*(.*)', paragraph)

    captions = [cap.strip().strip(', ').rstrip() for cap in captions if len(cap) > 5]
    n = min(len(timestamps), len(captions))
    return timestamps[:n], captions[:n]


def _dvc_format(caption: str) -> tuple[list[list[float]] | None, list[str] | None]:
    try:
        timestamps, sents = _extract_time_from_paragraph(caption)
    except Exception:
        return None, None

    if not timestamps:
        lines = caption.split('\n') if '\n' in caption else caption.split('.')
        lines = [line for line in lines if len(line) > 7]
        if '\n' not in caption:
            lines = [line + '.' for line in lines]
        for line in lines:
            if timestamps:
                break
            try:
                parts = [part.strip(',') for part in line.split('seconds')]
                time_part = _extract_time_part(parts[0])
                if not time_part:
                    continue
                start = round(float(time_part[0].split('-')[0].strip()), 2)
                end = round(float(time_part[0].split('-')[1].strip()), 2)
                timestamps.append([start, end])
                sents.append(parts[-1].strip())
            except Exception:
                continue

    if not timestamps:
        return None, None
    timestamps = [[min(span), max(span)] for span in timestamps]
    return timestamps, sents


def _is_prediction_file(path: str) -> bool:
    stem, ext = osp.splitext(osp.basename(path))
    if ext.lower() not in PRED_EXTS:
        return False
    if 'ETBench' not in stem:
        return False
    return not stem.endswith(EXCLUDED_STEM_SUFFIXES)


def find_prediction_files(root: str, data_filters: list[str] | None = None) -> list[str]:
    if osp.isfile(root):
        return [root] if _is_prediction_file(root) else []

    files = []
    for ext in PRED_EXTS:
        files.extend(glob.glob(osp.join(root, '**', f'*ETBench*{ext}'), recursive=True))

    out = []
    seen = set()
    for path in sorted(files):
        real = osp.realpath(path)
        if real in seen or not _is_prediction_file(path):
            continue
        if data_filters and not any(token in osp.basename(path) for token in data_filters):
            continue
        seen.add(real)
        out.append(path)
    return out


def _read_prediction_file(path: str) -> pd.DataFrame:
    ext = osp.splitext(path)[1].lower()
    if ext == '.xlsx':
        return pd.read_excel(path)
    if ext == '.csv':
        return pd.read_csv(path)
    if ext == '.tsv':
        return pd.read_csv(path, sep='\t')
    if ext == '.jsonl':
        records = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    if ext == '.json':
        with open(path, encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            if 'data' in obj and isinstance(obj['data'], list):
                return pd.DataFrame(obj['data'])
            return pd.DataFrame([obj])
    raise ValueError(f'Unsupported prediction file: {path}')


def _parse_answer(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _normalize_tgt(tgt: Any) -> list:
    if not tgt:
        return []
    if isinstance(tgt, tuple):
        tgt = list(tgt)
    if not isinstance(tgt, list):
        return []
    if tgt and isinstance(tgt[0], (int, float)):
        return [tgt]
    if tgt and isinstance(tgt[0], list) and tgt[0] and isinstance(tgt[0][0], list):
        return tgt[0]
    return tgt


def _build_samples(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if 'task' not in df.columns or 'prediction' not in df.columns:
        raise ValueError('Prediction file must contain task and prediction columns.')

    by_task: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        task = str(row.get('task', '')).lower().strip()
        if task not in set(GND_TASKS) | set(CAP_TASKS):
            continue

        gt = _parse_answer(row.get('answer', '{}'))
        sample = {
            'pred': str(row.get('prediction', '')).strip(),
            'task': task,
            'source': row.get('source', ''),
            'video': str(row.get('video', row.get('index', ''))),
            'tgt': _normalize_tgt(gt.get('tgt') or []),
            'g': gt.get('g') or [],
        }
        by_task.setdefault(task, []).append(sample)
    return by_task


def _dvc_fast_eval(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Compute only official DVC/SLC temporal F1 thresholds."""
    gt_dict: dict[str, dict[str, list]] = {}
    pred_dict: dict[str, list[dict[str, list[float]]]] = {}
    failed = 0

    for sample in samples:
        pred_spans, pred_caps = _dvc_format(sample['pred'])
        if pred_spans is None or pred_caps is None or not pred_spans:
            failed += 1
            continue
        vid = sample['video']
        gt_dict[vid] = {'timestamps': sample['tgt']}
        pred_dict[vid] = [{'timestamp': span} for span in pred_spans]

    out: dict[str, float] = {'Total': len(samples), 'Failed': failed}
    if not samples or not gt_dict:
        for thr in IOU_THRESHOLDS:
            out[f'F1@{thr}'] = 0.0
        out['F1'] = 0.0
        return out

    scale = len(pred_dict) / len(samples)
    f1_scores = []
    for thr_text in IOU_THRESHOLDS:
        thr = float(thr_text)
        precisions = []
        recalls = []
        for vid, gt in gt_dict.items():
            refs = gt['timestamps']
            preds = pred_dict.get(vid, [])
            ref_covered = set()
            pred_covered = set()
            for pred_idx, pred in enumerate(preds):
                for ref_idx, ref_ts in enumerate(refs):
                    if _iou_interval(pred['timestamp'], ref_ts) > thr:
                        ref_covered.add(ref_idx)
                        pred_covered.add(pred_idx)
            precisions.append(len(pred_covered) / len(preds) if preds else 0.0)
            recalls.append(len(ref_covered) / len(refs) if refs else 0.0)

        precision = sum(precisions) / len(precisions) * scale
        recall = sum(recalls) / len(recalls) * scale
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[f'F1@{thr_text}'] = round(f1, 5)
        f1_scores.append(f1)

    out['F1'] = round(sum(f1_scores) / len(f1_scores), 5)
    return out


def compute_fast_metrics(df: pd.DataFrame) -> dict[str, float]:
    by_task = _build_samples(df)
    collected: dict[str, dict[str, float]] = {}
    for task, samples in by_task.items():
        if task in ('tvg', 'epm'):
            collected[task] = _tvg_eval(samples)
        elif task == 'tal':
            collected[task] = _tal_eval(samples)
        elif task == 'evs':
            collected[task] = _evs_eval(samples)
        elif task == 'vhd':
            collected[task] = _vhd_eval(samples)
        elif task in CAP_TASKS:
            collected[task] = _dvc_fast_eval(samples)

    results: dict[str, float] = {}
    gnd_tasks = [task for task in GND_TASKS if task in collected]
    for task in gnd_tasks:
        data = collected[task]
        results[f'{task.upper()}/F1'] = round(data.get('F1', 0.0) * 100, 2)
        for thr in IOU_THRESHOLDS:
            key = f'F1@{thr}'
            if key in data:
                results[f'{task.upper()}/{key}'] = round(data[key] * 100, 2)
    if gnd_tasks:
        results['GND/F1'] = round(
            sum(collected[task].get('F1', 0.0) for task in gnd_tasks) / len(gnd_tasks) * 100, 2)
        for thr in IOU_THRESHOLDS:
            key = f'F1@{thr}'
            vals = [collected[task][key] for task in gnd_tasks if key in collected[task]]
            if vals:
                results[f'GND/{key}'] = round(sum(vals) / len(vals) * 100, 2)

    cap_tasks = [task for task in CAP_TASKS if task in collected]
    for task in cap_tasks:
        data = collected[task]
        results[f'{task.upper()}/F1'] = round(data.get('F1', 0.0) * 100, 2)
        for thr in IOU_THRESHOLDS:
            key = f'F1@{thr}'
            if key in data:
                results[f'{task.upper()}/{key}'] = round(data[key] * 100, 2)
    if cap_tasks:
        results['CAP/F1'] = round(
            sum(collected[task].get('F1', 0.0) for task in cap_tasks) / len(cap_tasks) * 100, 2)
        for thr in IOU_THRESHOLDS:
            key = f'F1@{thr}'
            vals = [collected[task][key] for task in cap_tasks if key in collected[task]]
            if vals:
                results[f'CAP/{key}'] = round(sum(vals) / len(vals) * 100, 2)

    return results


def _load_existing_score(score_file: str) -> dict[str, Any]:
    if not osp.exists(score_file):
        return {}
    with open(score_file, encoding='utf-8') as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _backup_once(path: str) -> None:
    if not osp.exists(path):
        return
    backup = path + '.bak'
    if not osp.exists(backup):
        shutil.copy2(path, backup)


def _write_score_files(pred_file: str, scores: dict[str, Any], backup: bool) -> tuple[str, str]:
    base = pred_file.rsplit('.', 1)[0]
    score_file = base + '_score.json'
    acc_file = base + '_etbench_acc.csv'

    if backup:
        _backup_once(score_file)
        _backup_once(acc_file)

    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)

    pd.DataFrame([{'metric': k, 'value': v} for k, v in scores.items()]).to_csv(
        acc_file, index=False)
    return score_file, acc_file


def patch_prediction_file(pred_file: str, backup: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
    df = _read_prediction_file(pred_file)
    fast_metrics = compute_fast_metrics(df)
    if not fast_metrics:
        raise ValueError(f'No ETBench GND/CAP samples found in {pred_file}')

    score_file = pred_file.rsplit('.', 1)[0] + '_score.json'
    scores = _load_existing_score(score_file)
    scores.update(fast_metrics)
    avg_keys = ('REF/Acc', 'GND/F1', 'CAP/F1', 'COM/mRec')
    if all(key in scores for key in avg_keys):
        scores['AVG'] = round(sum(float(scores[key]) for key in avg_keys) / len(avg_keys), 2)

    written_score, written_acc = _write_score_files(pred_file, scores, backup=backup)
    return scores, {'score': written_score, 'acc': written_acc}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Patch ETBench GND/CAP tIoU threshold metrics into existing score files.')
    parser.add_argument('paths', nargs='+', help='Prediction file(s) or directory/directories to scan')
    parser.add_argument('--data', nargs='*', default=None,
                        help='Optional filename substring filters, e.g. ETBench_adaptive')
    parser.add_argument('--dry-run', action='store_true', help='List files without writing')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create .bak files before overwriting score outputs')
    args = parser.parse_args()

    files = []
    for path in args.paths:
        files.extend(find_prediction_files(path, args.data))
    files = sorted(dict.fromkeys(files))

    if not files:
        print('No ETBench prediction files found.')
        return

    print(f'Found {len(files)} ETBench prediction file(s).')
    ok = 0
    for path in files:
        print(f'\n[ETBench patch] {path}')
        if args.dry_run:
            continue
        try:
            scores, written = patch_prediction_file(path, backup=not args.no_backup)
        except Exception as exc:
            print(f'  FAIL: {exc}')
            continue
        keys = [k for k in scores if k.startswith(('GND/', 'CAP/F1', 'CAP/'))]
        print(f'  wrote: {written["score"]}')
        print(f'  wrote: {written["acc"]}')
        print(f'  patched keys: {", ".join(sorted(keys))}')
        ok += 1

    if not args.dry_run:
        print(f'\nDone: patched {ok}/{len(files)} file(s).')


if __name__ == '__main__':
    main()
