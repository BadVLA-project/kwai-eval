"""Auto-discovery score loader — no hardcoded model or dataset registries."""

import glob
import json
import os

import numpy as np
import pandas as pd

from .config import COLORS, SCORE_FILE_PATTERNS


class ResultLoader:
    """Discover and load evaluation score files from work_dir."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self._models = []
        self._benchmarks = set()
        self._score_files = {}  # (model, benchmark) -> path
        self._model_colors = {}
        self._discover()

    # ── Auto-discovery ────────────────────────────────────────────────────

    def _discover(self):
        if not os.path.isdir(self.work_dir):
            return

        entries = sorted(os.listdir(self.work_dir))
        color_idx = 0

        for entry in entries:
            if entry.startswith('.') or entry.startswith('_'):
                continue
            model_dir = os.path.join(self.work_dir, entry)
            if not os.path.isdir(model_dir):
                continue

            # Collect all score files in this model dir (top-level + T*_G*/ subdirs)
            score_files = []
            for suffix, ext in SCORE_FILE_PATTERNS:
                pattern = os.path.join(model_dir, f'*{suffix}.{ext}')
                score_files.extend(glob.glob(pattern))
                pattern = os.path.join(model_dir, 'T*_G*', f'*{suffix}.{ext}')
                score_files.extend(glob.glob(pattern))

            if not score_files:
                continue

            self._models.append(entry)
            self._model_colors[entry] = COLORS[color_idx % len(COLORS)]
            color_idx += 1

            for fpath in score_files:
                fname = os.path.basename(fpath)
                bench = self._extract_benchmark(entry, fname)
                if bench:
                    self._benchmarks.add(bench)
                    key = (entry, bench)
                    if key not in self._score_files or fpath > self._score_files[key]:
                        self._score_files[key] = fpath

        self._benchmarks = sorted(self._benchmarks)

    def _extract_benchmark(self, model_dir, filename):
        """Extract benchmark name from filename by stripping model prefix and suffix."""
        name = filename
        for suffix, ext in SCORE_FILE_PATTERNS:
            ending = f'{suffix}.{ext}'
            if name.endswith(ending):
                name = name[:-len(ending)]
                break
        else:
            return None

        prefix = model_dir + '_'
        if name.startswith(prefix):
            name = name[len(prefix):]
        elif '_' in name:
            idx = name.find(model_dir)
            if idx >= 0:
                name = name[idx + len(model_dir) + 1:]

        return name if name else None

    # ── Public accessors ──────────────────────────────────────────────────

    @property
    def models(self):
        return list(self._models)

    @property
    def benchmarks(self):
        return list(self._benchmarks)

    def model_color(self, model):
        return self._model_colors.get(model, '#888888')

    # ── Score loading ─────────────────────────────────────────────────────

    def _load_file(self, path):
        if path.endswith('.csv'):
            return self._load_csv(path)
        elif path.endswith('.xlsx'):
            return self._load_xlsx(path)
        elif path.endswith('.json'):
            return self._load_json(path)
        return None

    @staticmethod
    def _load_csv(path):
        df = pd.read_csv(path)
        result = {}
        if 'category' in df.columns and 'accuracy' in df.columns:
            for _, row in df.iterrows():
                key = str(row.get('split', '')) + '/' + str(row['category']) \
                    if 'split' in df.columns and str(row.get('split', '')) not in ('Overall', 'nan', '') \
                    else str(row['category'])
                result[key] = float(row['accuracy'])
        return result

    @staticmethod
    def _load_xlsx(path):
        df = pd.read_excel(path)
        result = {}
        if 'category' in df.columns and 'accuracy' in df.columns:
            for _, row in df.iterrows():
                key = str(row.get('split', '')) + '/' + str(row['category']) \
                    if 'split' in df.columns and str(row.get('split', '')) not in ('Overall', 'nan', '') \
                    else str(row['category'])
                result[key] = float(row['accuracy'])
        return result

    @staticmethod
    def _load_json(path):
        with open(path) as f:
            return json.load(f)

    def _extract_primary(self, data, path):
        """Extract a single primary score (0-100) from raw data, auto-detecting format."""
        if data is None:
            return float('nan')

        # CSV/XLSX result: flat dict of {category: accuracy}
        if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            if 'overall' in data:
                return float(data['overall'])
            if 'Overall' in data:
                return float(data['Overall'])
            vals = [v for v in data.values() if isinstance(v, (int, float))]
            if vals:
                return sum(vals) / len(vals)
            return float('nan')

        if not isinstance(data, dict):
            return float('nan')

        # Video-MME-v2: has 'final_rating' dict
        if 'final_rating' in data:
            fr = data['final_rating']
            if isinstance(fr, dict) and 'total' in fr:
                return float(fr['total'])

        # Direct numeric 'overall'
        if 'overall' in data and isinstance(data['overall'], (int, float)):
            v = float(data['overall'])
            return v * 100 if v <= 1.0 else v

        # VideoMME v1: nested {duration: {overall: float}}
        if 'overall' in data and isinstance(data['overall'], dict):
            inner = data['overall']
            if 'overall' in inner:
                v = float(inner['overall'])
                return v * 100 if v <= 1.0 else v

        # Video-Holmes: {total: {acc: float}}
        if 'total' in data and isinstance(data['total'], dict) and 'acc' in data['total']:
            v = float(data['total']['acc'])
            return v * 100 if v <= 1.0 else v

        # MVBench: values are [correct, total, 'pct%'] lists
        if any(isinstance(v, list) and len(v) == 3 for v in data.values()):
            if 'overall' in data and isinstance(data['overall'], list):
                return float(str(data['overall'][2]).rstrip('%'))
            scores = []
            for v in data.values():
                if isinstance(v, list) and len(v) == 3:
                    scores.append(float(str(v[2]).rstrip('%')))
            if scores:
                return sum(scores) / len(scores)

        # Common keys
        for key in ('mIoU', 'total', 'accuracy', 'acc', 'score'):
            if key in data and isinstance(data[key], (int, float)):
                v = float(data[key])
                return v * 100 if v <= 1.0 else v

        # Fallback: average numeric top-level values
        nums = [float(v) for v in data.values() if isinstance(v, (int, float))]
        if nums:
            avg = sum(nums) / len(nums)
            return avg * 100 if avg <= 1.0 else avg

        return float('nan')

    def load_score(self, model, benchmark):
        """Load primary score (0-100) for a (model, benchmark) pair."""
        key = (model, benchmark)
        if key not in self._score_files:
            return float('nan')
        path = self._score_files[key]
        data = self._load_file(path)
        return self._extract_primary(data, path)

    def load_all_scores(self):
        """Return DataFrame: rows=models, cols=benchmarks, values=primary scores."""
        rows = {}
        for m in self._models:
            row = {}
            for b in self._benchmarks:
                row[b] = self.load_score(m, b)
            rows[m] = row
        if not self._benchmarks:
            return pd.DataFrame()
        return pd.DataFrame(rows).T[self._benchmarks]

    def load_breakdown(self, model, benchmark):
        """Load full score dict for sub-dimensions."""
        key = (model, benchmark)
        if key not in self._score_files:
            return None
        return self._load_file(self._score_files[key])
