"""Auto-discovery score loader — no hardcoded model or dataset registries."""

import glob
import json
import os

import numpy as np
import pandas as pd

from .config import COLORS, ETBENCH_GROUPS, MERGE_PREFIXES, SCORE_FILE_PATTERNS


class ResultLoader:
    """Discover and load evaluation score files from work_dir."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self._models = []
        self._benchmarks = set()
        self._score_files = {}  # (model, benchmark) -> path
        self._model_colors = {}
        self._model_configs = {}  # model -> config dict
        self._merged_data = {}
        self._discover()
        self._merge_prefixes()
        self._expand_etbench()

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
                pattern = os.path.join(model_dir, 'T*', f'*{suffix}.{ext}')
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

            # Load model config
            self._model_configs[entry] = self._load_model_config(model_dir)

        self._benchmarks = sorted(self._benchmarks)

    def _merge_prefixes(self):
        """Merge sub-benchmarks matching MERGE_PREFIXES into aggregate entries."""
        for prefix in MERGE_PREFIXES:
            sub_benches = [b for b in self._benchmarks if b.startswith(prefix + '_')]
            if len(sub_benches) < 2:
                continue

            for model in self._models:
                sub_scores = {}
                for sb in sub_benches:
                    key = (model, sb)
                    if key in self._score_files:
                        data = self._load_file(self._score_files[key])
                        primary = self._extract_primary(data, self._score_files[key])
                        if not np.isnan(primary):
                            sub_scores[sb.replace(prefix + '_', '')] = primary

                if sub_scores:
                    avg = sum(sub_scores.values()) / len(sub_scores)
                    self._merged_data[(model, prefix)] = {
                        'primary': avg,
                        'breakdown': sub_scores,
                    }

            for sb in sub_benches:
                self._benchmarks.remove(sb)
                for model in self._models:
                    self._score_files.pop((model, sb), None)
            self._benchmarks.append(prefix)

        self._benchmarks = sorted(self._benchmarks)

    def _expand_etbench(self):
        """Expand ETBench into sub-columns for each aggregation group."""
        if 'ETBench' not in self._benchmarks:
            return

        for model in self._models:
            key = (model, 'ETBench')
            if key not in self._score_files:
                continue

            data = self._load_file(self._score_files[key])
            if not isinstance(data, dict) or 'AVG' not in data:
                continue

            self._merged_data[key] = {
                'primary': float(data['AVG']),
                'breakdown': {g: float(data.get(cfg['key'], float('nan')))
                              for g, cfg in ETBENCH_GROUPS.items()},
            }

            for group_name, cfg in ETBENCH_GROUPS.items():
                sub_bench = f'ETBench/{group_name}'
                group_val = data.get(cfg['key'])
                if group_val is None:
                    continue
                sub_breakdown = {}
                for task_key in cfg['tasks']:
                    if task_key in data:
                        sub_breakdown[task_key] = float(data[task_key])
                self._merged_data[(model, sub_bench)] = {
                    'primary': float(group_val),
                    'breakdown': sub_breakdown,
                }

        has_etbench_data = any(
            (m, 'ETBench') in self._score_files for m in self._models
        )
        if has_etbench_data:
            for group_name in ETBENCH_GROUPS:
                sub_bench = f'ETBench/{group_name}'
                if sub_bench not in self._benchmarks:
                    self._benchmarks.append(sub_bench)

            for model in self._models:
                self._score_files.pop((model, 'ETBench'), None)

            self._benchmarks = sorted(self._benchmarks)

    # ── Public accessors ──────────────────────────────────────────────────

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

    @staticmethod
    def _load_model_config(model_dir: str) -> dict | None:
        """Load model_config.json from model_dir or its T* subdirs."""
        # Direct path
        direct = os.path.join(model_dir, 'model_config.json')
        if os.path.exists(direct):
            try:
                with open(direct) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        # T* subdir (pick the latest)
        candidates = sorted(glob.glob(os.path.join(model_dir, 'T*', 'model_config.json')))
        if candidates:
            try:
                with open(candidates[-1]) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    # ── Public accessors ──────────────────────────────────────────────────

    @property
    def models(self):
        return list(self._models)

    @property
    def benchmarks(self):
        return list(self._benchmarks)

    def model_color(self, model):
        return self._model_colors.get(model, '#888888')

    def model_config(self, model):
        return self._model_configs.get(model)

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
        elif 'metric' in df.columns and 'value' in df.columns:
            for _, row in df.iterrows():
                result[str(row['metric'])] = float(row['value'])
        elif 'task' in df.columns and 'acc' in df.columns:
            for _, row in df.iterrows():
                result[str(row['task'])] = float(row['acc'])
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
            if 'M-Avg' in data:
                return float(data['M-Avg'])
            if 'overall' in data:
                return float(data['overall'])
            if 'Overall' in data:
                return float(data['Overall'])
            if 'AVG' in data:
                return float(data['AVG'])
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
        merged = self._merged_data.get((model, benchmark))
        if merged is not None:
            return merged['primary']
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
        merged = self._merged_data.get((model, benchmark))
        if merged is not None:
            return merged['breakdown']
        key = (model, benchmark)
        if key not in self._score_files:
            return None
        data = self._load_file(self._score_files[key])
        if data is None:
            return None
        # Flatten nested dicts for display; keep only numeric leaf values
        flat = {}
        self._flatten(data, '', flat)
        return flat if flat else data

    @staticmethod
    def _flatten(obj, prefix, out):
        """Recursively flatten nested dicts into dot-separated keys with numeric values."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f'{prefix}{k}' if not prefix else f'{prefix}/{k}'
                if isinstance(v, (int, float)):
                    out[new_key] = round(float(v), 2)
                elif isinstance(v, dict):
                    ResultLoader._flatten(v, new_key, out)
                elif isinstance(v, list) and len(v) == 3:
                    # MVBench format: [correct, total, 'pct%']
                    try:
                        out[new_key] = float(str(v[2]).rstrip('%'))
                    except (ValueError, IndexError):
                        pass
