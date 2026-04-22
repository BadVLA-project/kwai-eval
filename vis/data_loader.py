"""Auto-discovery score loader with benchmark-aware table schemas."""

import glob
import json
import os

import numpy as np
import pandas as pd

from .config import (
    COLORS,
    ETBENCH_GROUPS,
    ETBENCH_TABLE_ORDER,
    GROUNDING_PRIMARY_KEYS,
    MERGE_PREFIXES,
    PRIMARY_METRIC_KEYS,
    SCORE_FILE_PATTERNS,
    VIDEO_MME_TABLE_COLUMNS,
    VINOGROUND_TABLE_COLUMNS,
)


class ResultLoader:
    """Discover evaluation results and map them into explicit display columns."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self._models = []
        self._benchmarks = set()
        self._score_files = {}  # (model, benchmark) -> path
        self._model_colors = {}
        self._model_configs = {}  # model -> config dict
        self._merged_data = {}
        self._data_cache = {}
        self._table_groups = []
        self._table_columns = []
        self._table_column_map = {}
        self._discover()
        self._merge_prefixes()
        self._expand_etbench()
        self._build_table_schema()

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
                    if key not in self._score_files:
                        continue
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
        """Expand ETBench into official group metrics for table display."""
        etbench_benches = []
        for bench in list(self._benchmarks):
            if not bench.upper().startswith('ETBENCH'):
                continue
            for model in self._models:
                key = (model, bench)
                if key not in self._score_files:
                    continue
                data = self._load_file(self._score_files[key])
                if isinstance(data, dict) and 'AVG' in data:
                    etbench_benches.append(bench)
                    break

        if not etbench_benches:
            return

        canonical = min(etbench_benches, key=len)

        for model in self._models:
            data = None
            for bench in etbench_benches:
                key = (model, bench)
                if key not in self._score_files:
                    continue
                candidate = self._load_file(self._score_files[key])
                if isinstance(candidate, dict) and 'AVG' in candidate:
                    data = candidate
                    if bench == canonical:
                        break

            if data is None:
                continue

            self._merged_data[(model, canonical)] = {
                'primary': float(data['AVG']),
                'breakdown': {
                    group_name: float(data.get(cfg['key'], float('nan')))
                    for group_name, cfg in ETBENCH_GROUPS.items()
                },
            }

            for group_name, cfg in ETBENCH_GROUPS.items():
                sub_bench = f'{canonical}/{group_name}'
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

        for bench in etbench_benches:
            if bench in self._benchmarks:
                self._benchmarks.remove(bench)
            for model in self._models:
                self._score_files.pop((model, bench), None)

        self._benchmarks.append(canonical)
        for group_name in ETBENCH_GROUPS:
            self._benchmarks.append(f'{canonical}/{group_name}')

        self._benchmarks = sorted(self._benchmarks)

    def _build_table_schema(self):
        """Build paper-style multi-column table groups."""
        groups = []
        columns = []

        for bench in self._benchmarks:
            if '/' in bench:
                continue

            child_benches = [b for b in self._benchmarks if b.startswith(f'{bench}/')]
            sample = self._sample_breakdown(bench)

            if bench.upper().startswith('ETBENCH') and child_benches:
                group_cols = []
                for label in ETBENCH_TABLE_ORDER:
                    if label == 'AVG':
                        group_cols.append({
                            'id': bench,
                            'benchmark': bench,
                            'label': 'AVG',
                            'chart_label': f'{bench}/AVG',
                            'metric_key': None,
                            'breakdown_source': bench,
                        })
                        continue

                    child_bench = f'{bench}/{label}'
                    if child_bench not in self._benchmarks:
                        continue
                    group_cols.append({
                        'id': child_bench,
                        'benchmark': child_bench,
                        'label': label,
                        'chart_label': child_bench,
                        'metric_key': None,
                        'breakdown_source': child_bench,
                    })

                if group_cols:
                    groups.append({
                        'id': bench,
                        'label': bench,
                        'summary_column_id': bench,
                        'columns': group_cols,
                    })
                    columns.extend(group_cols)
                continue

            if self._looks_like_vinoground(sample):
                group_cols = []
                for metric_key, label in VINOGROUND_TABLE_COLUMNS:
                    group_cols.append({
                        'id': f'{bench}::{metric_key}',
                        'benchmark': bench,
                        'label': label,
                        'chart_label': f'{bench}/{label}',
                        'metric_key': metric_key,
                        'breakdown_source': bench,
                    })
                groups.append({
                    'id': bench,
                    'label': bench,
                    'summary_column_id': f'{bench}::group_score',
                    'columns': group_cols,
                })
                columns.extend(group_cols)
                continue

            if self._looks_like_videomme(sample):
                group_cols = []
                for metric_key, label in VIDEO_MME_TABLE_COLUMNS:
                    group_cols.append({
                        'id': f'{bench}::{metric_key}',
                        'benchmark': bench,
                        'label': label,
                        'chart_label': f'{bench}/{label}',
                        'metric_key': metric_key,
                        'breakdown_source': bench,
                    })
                groups.append({
                    'id': bench,
                    'label': bench,
                    'summary_column_id': f'{bench}::overall/overall',
                    'columns': group_cols,
                })
                columns.extend(group_cols)
                continue

            metric_key = self._primary_key_for_metrics(sample, bench)
            single_col = {
                'id': bench,
                'benchmark': bench,
                'label': metric_key or 'Score',
                'chart_label': bench,
                'metric_key': metric_key,
                'breakdown_source': bench,
            }
            groups.append({
                'id': bench,
                'label': bench,
                'summary_column_id': bench,
                'columns': [single_col],
            })
            columns.append(single_col)

        self._table_groups = groups
        self._table_columns = columns
        self._table_column_map = {col['id']: col for col in columns}

    def _sample_breakdown(self, benchmark):
        for model in self._models:
            data = self.load_breakdown(model, benchmark)
            if isinstance(data, dict) and data:
                return data
        return None

    @staticmethod
    def _looks_like_vinoground(metrics):
        if not isinstance(metrics, dict):
            return False
        return all(key in metrics for key, _ in VINOGROUND_TABLE_COLUMNS)

    @staticmethod
    def _looks_like_videomme(metrics):
        if not isinstance(metrics, dict):
            return False
        return all(key in metrics for key, _ in VIDEO_MME_TABLE_COLUMNS)

    @staticmethod
    def _is_count_like_metric(key):
        lowered = str(key).strip().lower()
        return lowered in {'samples', 'sample', 'total', 'failed', 'count', 'num'}

    def _primary_key_for_metrics(self, metrics, benchmark=''):
        if not isinstance(metrics, dict) or not metrics:
            return None

        lower_bench = benchmark.lower()
        if any(token in lower_bench for token in ('ground', 'grounding', 'refcoco', 'charades', 'timelens')):
            for key in GROUNDING_PRIMARY_KEYS:
                if key in metrics:
                    return key

        for key in GROUNDING_PRIMARY_KEYS + PRIMARY_METRIC_KEYS:
            if key in metrics:
                return key

        numeric_keys = [
            key for key, value in metrics.items()
            if isinstance(value, (int, float)) and not self._is_count_like_metric(key)
        ]
        if len(numeric_keys) == 1:
            return numeric_keys[0]
        return None

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
        direct = os.path.join(model_dir, 'model_config.json')
        if os.path.exists(direct):
            try:
                with open(direct) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        candidates = sorted(glob.glob(os.path.join(model_dir, 'T*', 'model_config.json')))
        if candidates:
            try:
                with open(candidates[-1]) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    @property
    def models(self):
        return list(self._models)

    @property
    def benchmarks(self):
        return list(self._benchmarks)

    @property
    def table_groups(self):
        return list(self._table_groups)

    @property
    def table_columns(self):
        return list(self._table_columns)

    def model_color(self, model):
        return self._model_colors.get(model, '#888888')

    def model_config(self, model):
        return self._model_configs.get(model)

    # ── Score loading ─────────────────────────────────────────────────────

    def _load_file(self, path):
        if path in self._data_cache:
            return self._data_cache[path]

        if path.endswith('.csv'):
            data = self._load_csv(path)
        elif path.endswith('.xlsx'):
            data = self._load_xlsx(path)
        elif path.endswith('.json'):
            data = self._load_json(path)
        else:
            data = None

        self._data_cache[path] = data
        return data

    @classmethod
    def _load_csv(cls, path):
        df = pd.read_csv(path)
        return cls._frame_to_metric_dict(df)

    @classmethod
    def _load_xlsx(cls, path):
        df = pd.read_excel(path)
        return cls._frame_to_metric_dict(df)

    @classmethod
    def _frame_to_metric_dict(cls, df):
        result = {}

        if 'category' in df.columns and 'accuracy' in df.columns:
            for _, row in df.iterrows():
                key = str(row.get('split', '')) + '/' + str(row['category']) \
                    if 'split' in df.columns and str(row.get('split', '')) not in ('Overall', 'nan', '') \
                    else str(row['category'])
                result[key] = float(row['accuracy'])
            return result

        if 'metric' in df.columns and 'value' in df.columns:
            for _, row in df.iterrows():
                result[str(row['metric'])] = float(row['value'])
            return result

        if 'task' in df.columns and 'acc' in df.columns:
            for _, row in df.iterrows():
                result[str(row['task'])] = float(row['acc'])
            return result

        if 'Split' in df.columns:
            numeric_cols = []
            for col in df.columns:
                if col == 'Split' or cls._is_count_like_metric(col):
                    continue
                series = pd.to_numeric(df[col], errors='coerce')
                if series.notna().any():
                    numeric_cols.append(col)

            for _, row in df.iterrows():
                split = str(row['Split'])
                for col in numeric_cols:
                    try:
                        result[f'{split}/{col}'] = float(row[col])
                    except (TypeError, ValueError):
                        continue
            return result

        if len(df) == 1:
            row = df.iloc[0]
            for col in df.columns:
                if cls._is_count_like_metric(col):
                    continue
                try:
                    result[str(col)] = float(row[col])
                except (TypeError, ValueError):
                    continue
        return result

    @staticmethod
    def _load_json(path):
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _normalize_score(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float('nan')
        if np.isnan(v):
            return float('nan')
        return round(v * 100, 2) if abs(v) <= 1.0 else round(v, 2)

    def _extract_primary_from_flat(self, flat):
        """Extract a primary score from flattened nested metrics."""
        if not isinstance(flat, dict) or not flat:
            return float('nan')

        for key in GROUNDING_PRIMARY_KEYS + PRIMARY_METRIC_KEYS:
            if key in flat:
                return self._normalize_score(flat[key])

        nums = [
            float(value) for key, value in flat.items()
            if isinstance(value, (int, float)) and not self._is_count_like_metric(key)
        ]
        if nums:
            return self._normalize_score(sum(nums) / len(nums))
        return float('nan')

    def _extract_primary(self, data, path):
        """Extract a single primary score (0-100) from raw data."""
        if data is None:
            return float('nan')

        if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            for key in GROUNDING_PRIMARY_KEYS + PRIMARY_METRIC_KEYS:
                if key in data:
                    return self._normalize_score(data[key])
            if len(data) == 1:
                return self._normalize_score(next(iter(data.values())))
            vals = [v for v in data.values() if isinstance(v, (int, float))]
            if vals:
                return self._normalize_score(sum(vals) / len(vals))
            return float('nan')

        if not isinstance(data, dict):
            return float('nan')

        if 'final_rating' in data:
            fr = data['final_rating']
            if isinstance(fr, dict) and 'total' in fr:
                return self._normalize_score(fr['total'])

        if 'overall' in data and isinstance(data['overall'], (int, float)):
            return self._normalize_score(data['overall'])

        if 'overall' in data and isinstance(data['overall'], dict):
            inner = data['overall']
            if 'overall' in inner:
                return self._normalize_score(inner['overall'])
            for key in ('score', 'acc', 'accuracy', 'mIoU'):
                if key in inner:
                    return self._normalize_score(inner[key])

        if 'total' in data and isinstance(data['total'], dict) and 'acc' in data['total']:
            return self._normalize_score(data['total']['acc'])

        if any(isinstance(v, list) and len(v) == 3 for v in data.values()):
            if 'overall' in data and isinstance(data['overall'], list):
                return float(str(data['overall'][2]).rstrip('%'))
            scores = []
            for value in data.values():
                if isinstance(value, list) and len(value) == 3:
                    scores.append(float(str(value[2]).rstrip('%')))
            if scores:
                return sum(scores) / len(scores)

        if any(isinstance(v, list) and len(v) == 2 for v in data.values()):
            total_correct = 0
            total_count = 0
            for value in data.values():
                if isinstance(value, list) and len(value) == 2:
                    total_correct += value[0]
                    total_count += value[1]
            if total_count > 0:
                return round(total_correct / total_count * 100, 2)

        for key in GROUNDING_PRIMARY_KEYS + PRIMARY_METRIC_KEYS + ['total']:
            if key in data and isinstance(data[key], (int, float)):
                return self._normalize_score(data[key])

        flat = {}
        self._flatten(data, '', flat)
        flat_primary = self._extract_primary_from_flat(flat)
        if not np.isnan(flat_primary):
            return flat_primary

        nums = [float(v) for v in data.values() if isinstance(v, (int, float))]
        if nums:
            return self._normalize_score(sum(nums) / len(nums))

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
        for model in self._models:
            rows[model] = {bench: self.load_score(model, bench) for bench in self._benchmarks}
        if not self._benchmarks:
            return pd.DataFrame()
        return pd.DataFrame(rows).T[self._benchmarks]

    def load_column_score(self, model, column_id):
        """Load the score shown in one visible table column."""
        spec = self._table_column_map.get(column_id)
        if spec is None:
            return float('nan')

        metric_key = spec.get('metric_key')
        if metric_key is None:
            return self.load_score(model, spec['benchmark'])

        breakdown = self.load_breakdown(model, spec['benchmark'])
        if not isinstance(breakdown, dict) or metric_key not in breakdown:
            return self.load_score(model, spec['benchmark'])
        value = self._normalize_score(breakdown[metric_key])
        if np.isnan(value):
            return self.load_score(model, spec['benchmark'])
        return value

    def load_all_column_scores(self):
        """Return DataFrame: rows=models, cols=visible table columns."""
        col_ids = [col['id'] for col in self._table_columns]
        rows = {}
        for model in self._models:
            rows[model] = {col_id: self.load_column_score(model, col_id) for col_id in col_ids}
        if not col_ids:
            return pd.DataFrame()
        return pd.DataFrame(rows).T[col_ids]

    def load_column_breakdown(self, model, column_id):
        """Load breakdown values for one visible table column."""
        spec = self._table_column_map.get(column_id)
        if spec is None:
            return None

        source = spec.get('breakdown_source') or spec['benchmark']
        data = self.load_breakdown(model, source)
        if not isinstance(data, dict) or not data:
            return None

        normalized = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                normalized[key] = self._normalize_score(value)

        if len(normalized) <= 1:
            return None
        return normalized

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
        flat = {}
        self._flatten(data, '', flat)
        return flat if flat else data

    @staticmethod
    def _flatten(obj, prefix, out):
        """Recursively flatten nested dicts into slash-separated numeric keys."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f'{prefix}{key}' if not prefix else f'{prefix}/{key}'
                if isinstance(value, (int, float)):
                    out[new_key] = float(value)
                elif isinstance(value, str):
                    try:
                        out[new_key] = float(value)
                    except ValueError:
                        pass
                elif isinstance(value, dict):
                    ResultLoader._flatten(value, new_key, out)
                elif isinstance(value, list) and len(value) == 3:
                    try:
                        out[new_key] = float(str(value[2]).rstrip('%'))
                    except (ValueError, IndexError):
                        pass
                elif isinstance(value, list) and len(value) == 2:
                    try:
                        correct, total = int(value[0]), int(value[1])
                        out[new_key] = (correct / total * 100) if total > 0 else 0.0
                    except (ValueError, ZeroDivisionError, TypeError):
                        pass
