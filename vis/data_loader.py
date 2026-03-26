"""Unified score file discovery and loading for all benchmark datasets."""

import glob
import json
import os
import re

import pandas as pd
import numpy as np

from .config import (
    MODEL_NAMES, DATASET_NAMES, DATASET_INFO, DATASET_LABELS,
    MODEL_LABELS, PRIMARY_METRIC, AOT_DATASETS, BASE_MODEL,
    OVERALL_BENCHMARKS, MODEL_INFO,
)

# Colour palette for auto-discovered models (cycles if needed)
_AUTO_COLORS = [
    '#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#1abc9c',
    '#3498db', '#9b59b6', '#8e44ad', '#34495e', '#16a085',
    '#2980b9', '#c0392b', '#d35400', '#7f8c8d', '#2c3e50',
    '#6c5ce7', '#00b894', '#fd79a8', '#fdcb6e', '#636e72',
]


class ResultLoader:
    """Discover and load evaluation score files from WORK_DIR."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        # Auto-discover models not in MODEL_INFO
        self._extra_models = self._discover_extra_models()

    def _discover_extra_models(self):
        """Scan work_dir for model directories not registered in MODEL_INFO.

        A directory is treated as a model dir only if it contains at least one
        known result file pattern (e.g. *_acc.csv, *_rating.json, *_score.json).
        This prevents dataset output dirs, source-code dirs, etc. from appearing.
        """
        if not os.path.isdir(self.work_dir):
            return {}
        known_keys = set(MODEL_INFO.keys())
        known_datasets = set(DATASET_NAMES)

        # Patterns that indicate a directory is a model result dir
        _RESULT_PATTERNS = ('*_acc.csv', '*_rating.json', '*_score.json', '*_acc.xlsx')

        extra = {}
        try:
            entries = sorted(os.listdir(self.work_dir))
        except OSError:
            return {}

        # Build a set of used labels to ensure uniqueness
        used_labels = {v[0] for v in MODEL_INFO.values()}
        color_idx = len(MODEL_INFO)

        for entry in entries:
            if entry in known_keys or entry in known_datasets:
                continue
            if entry.startswith('.') or entry.startswith('_'):
                continue
            full = os.path.join(self.work_dir, entry)
            if not os.path.isdir(full):
                continue

            # Must contain at least one result file at top level or in T*_G*/ subdirs
            has_result = False
            for pat in _RESULT_PATTERNS:
                if glob.glob(os.path.join(full, pat)):
                    has_result = True
                    break
                if glob.glob(os.path.join(full, 'T*_G*', pat)):
                    has_result = True
                    break
            if not has_result:
                continue

            # Build a unique, readable label from the directory name
            label = entry
            # If label is already taken, append a counter suffix
            if label in used_labels:
                i = 2
                while f'{label}_{i}' in used_labels:
                    i += 1
                label = f'{label}_{i}'
            used_labels.add(label)

            color = _AUTO_COLORS[color_idx % len(_AUTO_COLORS)]
            color_idx += 1
            extra[entry] = (label, 'extra', color)
        return extra

    def get_all_model_info(self):
        """Return merged OrderedDict of MODEL_INFO + auto-discovered models."""
        from collections import OrderedDict
        merged = OrderedDict(MODEL_INFO)
        merged.update(self._extra_models)
        return merged

    def get_all_model_names(self):
        return list(self.get_all_model_info().keys())

    def get_model_labels(self):
        return {k: v[0] for k, v in self.get_all_model_info().items()}

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

    def _model_label(self, model_key):
        """Get display label for a model, supporting both registered and auto-discovered."""
        if model_key in MODEL_LABELS:
            return MODEL_LABELS[model_key]
        if model_key in self._extra_models:
            return self._extra_models[model_key][0]
        return model_key

    def load_overall_matrix(self, models=None):
        """Return DataFrame: rows=models, cols=7 benchmarks (AoT averaged), values=0-100."""
        models = models or self.get_all_model_names()
        non_aot = [d for d in DATASET_NAMES if d not in AOT_DATASETS]

        data = {}
        for m in models:
            row = {}
            # AoTBench: average of 5 sub-benchmarks
            aot_scores = [self.get_primary_score(m, d) for d in AOT_DATASETS]
            aot_valid = [s for s in aot_scores if not np.isnan(s)]
            row['AoTBench'] = np.mean(aot_valid) if aot_valid else float('nan')
            # Other benchmarks
            for d in non_aot:
                row[DATASET_LABELS[d]] = self.get_primary_score(m, d)
            data[self._model_label(m)] = row

        # Ensure column order matches OVERALL_BENCHMARKS
        df = pd.DataFrame(data).T
        cols = [c for c in OVERALL_BENCHMARKS if c in df.columns]
        return df[cols]

    def load_videomme_duration(self, models=None):
        """Return DataFrame: rows=models, cols=[short, medium, long, overall]."""
        models = models or self.get_all_model_names()
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'Video-MME_64frame')
            lbl = self._model_label(m)
            if data is None:
                rows[lbl] = {d: float('nan') for d in ['short', 'medium', 'long', 'overall']}
            else:
                rows[lbl] = {
                    d: data.get(d, {}).get('overall', float('nan'))
                    for d in ['short', 'medium', 'long', 'overall']
                }
        return pd.DataFrame(rows).T

    def load_mvbench_tasks(self, models=None):
        """Return DataFrame: rows=models, cols=20 MVBench sub-tasks."""
        models = models or self.get_all_model_names()
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'MVBench_MP4_1fps')
            lbl = self._model_label(m)
            if data is None:
                rows[lbl] = {}
            else:
                rows[lbl] = {k: v for k, v in data.items() if k != 'overall'}
        return pd.DataFrame(rows).T

    def load_videoholmes_types(self, models=None):
        """Return DataFrame: rows=models, cols=7 question types."""
        models = models or self.get_all_model_names()
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'Video_Holmes_64frame')
            lbl = self._model_label(m)
            if data is None:
                rows[lbl] = {}
            else:
                rows[lbl] = {k: v for k, v in data.items() if k != 'total'}
        return pd.DataFrame(rows).T

    def load_videomme_tasktype(self, models=None):
        """Return DataFrame: rows=models, cols=12 VideoMME task types."""
        models = models or self.get_all_model_names()
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'Video-MME_64frame')
            lbl = self._model_label(m)
            if data is None:
                rows[lbl] = {}
            else:
                overall = data.get('overall', {})
                rows[lbl] = overall.get('task_type', {})
        return pd.DataFrame(rows).T

    def load_perception_dims(self, models=None):
        """Return dict of DataFrames: {split: DataFrame(rows=models, cols=categories)}."""
        models = models or self.get_all_model_names()
        splits = {}
        for m in models:
            data = self.load_dataset_score(m, 'PerceptionTest_val_16frame')
            if data is None:
                continue
            lbl = self._model_label(m)
            for key, val in data.items():
                if key == 'Overall':
                    continue
                if '/' in key:
                    split, cat = key.split('/', 1)
                    splits.setdefault(split, {}).setdefault(lbl, {})[cat] = val
        return {s: pd.DataFrame(d).T for s, d in splits.items()}

    def load_charades_metrics(self, models=None):
        """Return DataFrame: rows=models, cols=[mIoU, R@1_IoU=0.3, R@1_IoU=0.5, R@1_IoU=0.7]."""
        models = models or self.get_all_model_names()
        rows = {}
        for m in models:
            data = self.load_dataset_score(m, 'CharadesTimeLens_1fps')
            lbl = self._model_label(m)
            if data is None:
                rows[lbl] = {}
            else:
                rows[lbl] = data
        return pd.DataFrame(rows).T

    def load_aot_subsets(self, models=None):
        """Return DataFrame: rows=models, cols=5 AoT sub-benchmarks, values=accuracy 0-100."""
        models = models or self.get_all_model_names()
        rows = {}
        for m in models:
            row = {}
            for d in AOT_DATASETS:
                row[DATASET_LABELS[d]] = self.get_primary_score(m, d)
            rows[self._model_label(m)] = row
        return pd.DataFrame(rows).T

    # ── Completeness checks ──────────────────────────────────────────────

    def check_completeness(self, models=None):
        """Check evaluation completeness for each (model, dataset) pair.

        Returns a dict:
            {
                model_key: {
                    'label': str,
                    'group': str,
                    'color': str,
                    'datasets': {
                        dataset_key: {
                            'label': str,
                            'done': bool,
                            'path': str or None,
                        }
                    },
                    'complete_count': int,
                    'total_count': int,
                    'pct': float,   # 0-100
                }
            }
        """
        all_info = self.get_all_model_info()
        models = models or list(all_info.keys())
        result = {}
        for m in models:
            info = all_info.get(m, (m, 'extra', '#888'))
            label, group, color = info
            datasets = {}
            complete = 0
            for d in DATASET_NAMES:
                score_type = DATASET_INFO[d][1]
                suffix, ext = self._SUFFIX_EXT[score_type]
                if score_type == 'perception_acc':
                    path = self._find_file(m, d, suffix, 'xlsx')
                    if path is None:
                        path = self._find_file(m, d, suffix, 'csv')
                else:
                    path = self._find_file(m, d, suffix, ext)
                done = path is not None
                if done:
                    complete += 1
                datasets[d] = {
                    'label': DATASET_LABELS[d],
                    'done': done,
                    'path': path,
                }
            total = len(DATASET_NAMES)
            result[m] = {
                'label': label,
                'group': group,
                'color': color,
                'datasets': datasets,
                'complete_count': complete,
                'total_count': total,
                'pct': round(complete / total * 100, 1) if total else 0.0,
            }
        return result
