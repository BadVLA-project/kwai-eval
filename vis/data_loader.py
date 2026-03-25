"""Unified score file discovery and loading for all benchmark datasets."""

import glob
import json
import os

import pandas as pd
import numpy as np

from .config import (
    MODEL_NAMES, DATASET_NAMES, DATASET_INFO, DATASET_LABELS,
    MODEL_LABELS, PRIMARY_METRIC, AOT_DATASETS, BASE_MODEL,
)


class ResultLoader:
    """Discover and load evaluation score files from WORK_DIR."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir

    # ── File discovery ───────────────────────────────────────────────────

    def _find_file(self, model, dataset, suffix, ext):
        """Find score file by priority: model-level symlink → glob T*_G*/."""
        base = f'{model}_{dataset}{suffix}.{ext}'

        # 1) model-centric symlink
        p1 = os.path.join(self.work_dir, model, base)
        if os.path.exists(p1):
            return p1

        # 2) bench-centric symlink
        p2 = os.path.join(self.work_dir, dataset, model, base)
        if os.path.exists(p2):
            return p2

        # 3) glob inside T*_G*/ directories
        pattern = os.path.join(self.work_dir, model, 'T*_G*', base)
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]  # latest run

        return None

    # ── Per-format loaders ───────────────────────────────────────────────

    def _load_acc_csv(self, path):
        """Load AoTBench/FutureOmni _acc.csv → {category: accuracy}."""
        df = pd.read_csv(path)
        return {row['category']: float(row['accuracy'])
                for _, row in df.iterrows()}

    def _load_perception_acc(self, path):
        """Load PerceptionTest _acc.csv/.xlsx → {split/category: accuracy}."""
        if path.endswith('.xlsx'):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        result = {}
        for _, row in df.iterrows():
            key = f"{row['split']}/{row['category']}" if row['split'] != 'Overall' else 'Overall'
            result[key] = float(row['accuracy'])
        return result

    def _load_charades_json(self, path):
        """Load CharadesTimeLens _score.json → {metric: value}."""
        with open(path) as f:
            return json.load(f)  # already 0-100

    def _load_mvbench_rating(self, path):
        """Load MVBench _rating.json → {task: pct, ..., overall: pct}."""
        with open(path) as f:
            data = json.load(f)
        result = {}
        for task, val in data.items():
            if isinstance(val, list) and len(val) == 3:
                result[task] = float(str(val[2]).rstrip('%'))
            elif isinstance(val, (int, float)):
                result[task] = float(val)
        return result

    def _load_videomme_rating(self, path):
        """Load VideoMME _rating.json → nested dict with values as 0-100 floats."""
        with open(path) as f:
            data = json.load(f)
        result = {}
        for duration in ('short', 'medium', 'long', 'overall'):
            if duration not in data:
                continue
            d = data[duration]
            entry = {}
            # overall
            if 'overall' in d:
                entry['overall'] = float(d['overall']) * 100
            # domain, task_type, sub_category
            for dim in ('domain', 'task_type', 'sub_category'):
                if dim in d and isinstance(d[dim], dict):
                    entry[dim] = {k: float(v) * 100 for k, v in d[dim].items()}
            result[duration] = entry
        return result

    def _load_videoholmes_rating(self, path):
        """Load Video_Holmes _rating.json → {type: acc_pct, ..., total: acc_pct}."""
        with open(path) as f:
            data = json.load(f)
        result = {}
        if 'acc_by_type' in data:
            for qtype, info in data['acc_by_type'].items():
                result[qtype] = float(info['acc']) * 100
        if 'total' in data:
            result['total'] = float(data['total']['acc']) * 100
        return result

    # ── Dispatch loader ──────────────────────────────────────────────────

    _SUFFIX_EXT = {
        'acc_csv':           ('_acc', 'csv'),
        'perception_acc':    ('_acc', 'csv'),
        'charades_json':     ('_score', 'json'),
        'mvbench_rating':    ('_rating', 'json'),
        'videomme_rating':   ('_rating', 'json'),
        'videoholmes_rating': ('_rating', 'json'),
    }

    _LOADER_FN = {
        'acc_csv':           '_load_acc_csv',
        'perception_acc':    '_load_perception_acc',
        'charades_json':     '_load_charades_json',
        'mvbench_rating':    '_load_mvbench_rating',
        'videomme_rating':   '_load_videomme_rating',
        'videoholmes_rating': '_load_videoholmes_rating',
    }

    def load_dataset_score(self, model, dataset):
        """Load full score dict for a (model, dataset) pair. Returns None if not found."""
        score_type = DATASET_INFO[dataset][1]
        suffix, ext = self._SUFFIX_EXT[score_type]

        # perception_acc: try xlsx first, then csv
        if score_type == 'perception_acc':
            path = self._find_file(model, dataset, suffix, 'xlsx')
            if path is None:
                path = self._find_file(model, dataset, suffix, 'csv')
        else:
            path = self._find_file(model, dataset, suffix, ext)

        if path is None:
            return None

        loader = getattr(self, self._LOADER_FN[score_type])
        try:
            return loader(path)
        except Exception as e:
            print(f'  WARN: failed to load {path}: {e}')
            return None

    # ── Extract primary metric ───────────────────────────────────────────

    def get_primary_score(self, model, dataset):
        """Return the single primary metric (0-100) for overall comparison."""
        data = self.load_dataset_score(model, dataset)
        if data is None:
            return float('nan')

        metric = PRIMARY_METRIC[dataset]

        if metric == 'overall_accuracy':
            return data.get('overall', float('nan'))
        elif metric == 'mIoU':
            return data.get('mIoU', float('nan'))
        elif metric == 'overall_pct':
            return data.get('overall', float('nan'))
        elif metric == 'overall_overall':
            # VideoMME: data['overall']['overall']
            if 'overall' in data and isinstance(data['overall'], dict):
                return data['overall'].get('overall', float('nan'))
            return float('nan')
        elif metric == 'total_acc':
            return data.get('total', float('nan'))
        return float('nan')

    # ── Bulk loaders for plotting ────────────────────────────────────────

    def load_overall_matrix(self, models=None, datasets=None):
        """Return DataFrame: rows=models, cols=datasets, values=primary score."""
        models = models or MODEL_NAMES
        datasets = datasets or DATASET_NAMES
        data = {}
        for m in models:
            row = {}
            for d in datasets:
                row[DATASET_LABELS[d]] = self.get_primary_score(m, d)
            data[MODEL_LABELS[m]] = row
        return pd.DataFrame(data).T

    def load_videomme_duration(self, models=None):
        """Return DataFrame: rows=models, cols=[short, medium, long, overall]."""
        models = models or MODEL_NAMES
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'Video-MME_64frame')
            if data is None:
                rows[MODEL_LABELS[m]] = {d: float('nan') for d in ['short', 'medium', 'long', 'overall']}
            else:
                rows[MODEL_LABELS[m]] = {
                    d: data.get(d, {}).get('overall', float('nan'))
                    for d in ['short', 'medium', 'long', 'overall']
                }
        return pd.DataFrame(rows).T

    def load_mvbench_tasks(self, models=None):
        """Return DataFrame: rows=models, cols=20 MVBench sub-tasks."""
        models = models or MODEL_NAMES
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'MVBench_MP4_1fps')
            if data is None:
                rows[MODEL_LABELS[m]] = {}
            else:
                rows[MODEL_LABELS[m]] = {k: v for k, v in data.items() if k != 'overall'}
        return pd.DataFrame(rows).T

    def load_videoholmes_types(self, models=None):
        """Return DataFrame: rows=models, cols=7 question types."""
        models = models or MODEL_NAMES
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'Video_Holmes_64frame')
            if data is None:
                rows[MODEL_LABELS[m]] = {}
            else:
                rows[MODEL_LABELS[m]] = {k: v for k, v in data.items() if k != 'total'}
        return pd.DataFrame(rows).T

    def load_videomme_tasktype(self, models=None):
        """Return DataFrame: rows=models, cols=12 VideoMME task types."""
        models = models or MODEL_NAMES
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'Video-MME_64frame')
            if data is None:
                rows[MODEL_LABELS[m]] = {}
            else:
                overall = data.get('overall', {})
                rows[MODEL_LABELS[m]] = overall.get('task_type', {})
        return pd.DataFrame(rows).T

    def load_perception_dims(self, models=None):
        """Return dict of DataFrames: {split: DataFrame(rows=models, cols=categories)}."""
        models = models or MODEL_NAMES
        splits = {}
        for m in models:
            data = self.load_dataset_score(m, 'PerceptionTest_val_16frame')
            if data is None:
                continue
            for key, val in data.items():
                if key == 'Overall':
                    continue
                if '/' in key:
                    split, cat = key.split('/', 1)
                    splits.setdefault(split, {}).setdefault(MODEL_LABELS[m], {})[cat] = val
        return {s: pd.DataFrame(d).T for s, d in splits.items()}
