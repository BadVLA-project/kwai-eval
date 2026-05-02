#!/usr/bin/env python3
"""Build a reusable pairwise case-analysis bundle from row-level score files.

The goal is to turn raw benchmark outputs into a compact system for answering:
1. where the score moved at dataset level;
2. which subgroup concentrates the improvement or regression;
3. which concrete cases best explain the change.

The script intentionally works on existing row-level `_score` artifacts and
exports both human-readable summaries and a normalized case inventory.

Example:
    python scripts/build_case_report.py \
        --work-dir /path/to/eval_outputs \
        --baseline Qwen3-VL-4B-Instruct_direct \
        --candidate Qwen3-VL-4B-Instruct_direct_v2 \
        --out-dir ./case_report
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import os.path as osp
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('MPLCONFIGDIR', osp.join(tempfile.gettempdir(), 'mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME', tempfile.gettempdir())

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_PRIORITY = [
    'category',
    'source',
    'split',
    'area',
    'reasoning',
    'tag',
    'task',
    'task_type',
    'question_type',
    'subtask',
    'sub_task',
    'subcategory',
    'sub_category',
    'subject',
    'domain',
    'type',
]

EXCLUDED_GROUP_COLUMNS = {
    'index',
    'question_id',
    'id',
    'question',
    'answer',
    'prediction',
    'thinking',
    'extra_records',
    'image',
    'image_path',
    'video',
    'video_name',
    'video_path',
    'frames',
    'frame_paths',
    'candidates',
    'choices',
    'extracted_answer',
    'pred_id',
    'score',
    'correct',
    'iou',
    'metric_base',
    'metric_cand',
    '_case_id',
    'case_delta',
    'comparison',
    'case_bucket',
}

DISPLAY_COLUMNS = [
    'question',
    'answer',
    'category',
    'source',
    'split',
    'area',
    'reasoning',
    'tag',
    'video',
    'video_name',
    'video_path',
    'image_path',
]

COLORS = {
    'baseline': '#4C72B0',
    'candidate': '#DD8452',
    'gain': '#55A868',
    'regression': '#C44E52',
    'neutral': '#8C8C8C',
}


@dataclass
class MetricSpec:
    key: str
    kind: str
    scale: float


@dataclass
class DatasetSummary:
    dataset: str
    baseline_score: float
    candidate_score: float
    delta: float
    sample_count: int
    metric_key: str
    candidate_wins: int
    baseline_wins: int
    ties: int
    swing_rate: float
    stable_correct: int | None
    stable_wrong: int | None


@dataclass
class GroupSummary:
    dataset: str
    group_column: str
    group_value: str
    label: str
    baseline_score: float
    candidate_score: float
    delta: float
    sample_count: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    swing_rate: float
    stable_correct: int | None
    stable_wrong: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a pairwise case-analysis bundle')
    parser.add_argument('--work-dir', required=True, help='Evaluation result root directory')
    parser.add_argument('--baseline', required=True, help='Baseline model name')
    parser.add_argument('--candidate', required=True, help='Candidate model name')
    parser.add_argument('--out-dir', required=True, help='Output directory for charts and exports')
    parser.add_argument('--data', nargs='*', default=None, help='Optional dataset filter (exact names)')
    parser.add_argument(
        '--group-columns',
        nargs='*',
        default=None,
        help='Optional explicit subgroup columns; falls back to auto-discovery',
    )
    parser.add_argument('--min-group-size', type=int, default=8, help='Minimum subgroup size')
    parser.add_argument('--top-datasets', type=int, default=12, help='Max datasets shown in charts')
    parser.add_argument('--top-groups', type=int, default=10, help='Max positive/negative groups in report')
    parser.add_argument('--cases-per-group', type=int, default=3, help='Representative cases per group/direction')
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_file(path: str) -> Any:
    suffix = osp.splitext(path)[1].lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    if suffix == '.xlsx':
        return pd.read_excel(path)
    if suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    if suffix == '.jsonl':
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f'Unsupported file type: {path}')


def dump_json(obj: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write('\n')


def dump_table(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _to_dataframe(obj: Any) -> pd.DataFrame | None:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in obj):
            return pd.DataFrame(obj)
        return None
    if isinstance(obj, dict):
        if 'columns' in obj and 'data' in obj:
            return pd.DataFrame(obj['data'], columns=obj['columns'])
        values = list(obj.values())
        if values and all(isinstance(v, list) for v in values):
            lengths = {len(v) for v in values}
            if len(lengths) == 1:
                return pd.DataFrame(obj)
    return None


def load_row_level_score(path: str) -> pd.DataFrame | None:
    try:
        data = load_file(path)
    except Exception:
        return None
    df = _to_dataframe(data)
    if df is None or df.empty:
        return None
    id_col = next((c for c in ['index', 'question_id', 'id'] if c in df.columns), None)
    if id_col is None:
        return None
    df = df.copy()
    df['_case_id'] = df[id_col].astype(str)
    return df


def infer_metric_spec(df: pd.DataFrame) -> MetricSpec | None:
    if 'iou' in df.columns:
        return MetricSpec('iou', 'continuous', 100.0)
    if 'correct' in df.columns:
        return MetricSpec('correct', 'binary', 100.0)
    if 'score' in df.columns:
        vals = pd.to_numeric(df['score'], errors='coerce').dropna()
        if len(vals) == 0:
            return None
        uniq = set(vals.unique().tolist())
        if uniq.issubset({-1, 0, 1}) or uniq.issubset({0, 1}):
            return MetricSpec('score', 'binary', 100.0)
        scale = 100.0 if vals.min() >= 0 and vals.max() <= 1.0 else 1.0
        return MetricSpec('score', 'continuous', scale)
    return None


def metric_series(df: pd.DataFrame, spec: MetricSpec) -> pd.Series:
    if spec.key == 'correct':
        return df[spec.key].astype(float)
    return pd.to_numeric(df[spec.key], errors='coerce')


def valid_metric_mask(values: pd.Series, spec: MetricSpec) -> pd.Series:
    valid = values.notna()
    if spec.kind in {'binary', 'continuous'}:
        valid &= values >= 0
    return valid


def aggregate_score(values: pd.Series, spec: MetricSpec) -> float:
    valid = values[valid_metric_mask(values, spec)]
    if len(valid) == 0:
        return float('nan')
    return float(valid.mean() * spec.scale)


def shorten(text: Any, limit: int = 180) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ''
    text = str(text).strip().replace('\n', ' ')
    text = ' '.join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + '...'


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def benchmark_from_path(model: str, path: str) -> str | None:
    stem = osp.splitext(osp.basename(path))[0]
    prefix = f'{model}_'
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix):]
    if not tail.endswith('_score'):
        return None
    return tail[:-len('_score')]


def model_score_files(work_dir: str, model: str) -> dict[str, str]:
    model_dir = osp.join(work_dir, model)
    if not osp.isdir(model_dir):
        raise FileNotFoundError(f'Model directory not found: {model_dir}')

    patterns = [
        osp.join(model_dir, f'{model}_*_score.*'),
        osp.join(model_dir, 'T*', f'{model}_*_score.*'),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    grouped: dict[str, list[str]] = {}
    for path in files:
        bench = benchmark_from_path(model, path)
        if bench is None:
            continue
        grouped.setdefault(bench, []).append(path)

    resolved: dict[str, str] = {}
    for bench, candidates in grouped.items():
        usable = []
        for path in sorted(candidates):
            df = load_row_level_score(path)
            if df is None:
                continue
            spec = infer_metric_spec(df)
            if spec is None:
                continue
            usable.append(path)
        if usable:
            resolved[bench] = sorted(usable)[-1]
    return resolved


def coalesce_columns(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    df = df.copy()
    for name in names:
        cand_col = f'{name}_cand'
        base_col = f'{name}_base'
        if cand_col in df.columns and base_col in df.columns:
            df[name] = df[cand_col].combine_first(df[base_col])
        elif cand_col in df.columns:
            df[name] = df[cand_col]
        elif base_col in df.columns:
            df[name] = df[base_col]
    return df


def discover_group_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if col in EXCLUDED_GROUP_COLUMNS:
            continue
        if col.endswith('_base') or col.endswith('_cand'):
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        nunique = series.astype(str).nunique()
        if nunique < 2 or nunique > 12:
            continue
        if nunique > len(series) * 0.5:
            continue
        if series.astype(str).map(len).mean() > 40:
            continue
        candidates.append(col)

    ordered = [c for c in GROUP_PRIORITY if c in candidates]
    leftovers = sorted([c for c in candidates if c not in ordered], key=lambda x: (df[x].nunique(), x))
    return ordered + leftovers


def resolve_group_columns(df: pd.DataFrame, explicit_columns: list[str] | None) -> list[str]:
    if explicit_columns:
        return [col for col in explicit_columns if col in df.columns]
    return discover_group_columns(df)


def build_pair_dataframe(
    base_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    metric: MetricSpec,
) -> pd.DataFrame:
    base = base_df.copy()
    cand = cand_df.copy()
    mergeable_columns = sorted(
        {
            c
            for c in list(base.columns) + list(cand.columns)
            if c not in EXCLUDED_GROUP_COLUMNS and not c.endswith('_base') and not c.endswith('_cand')
        }
    )

    base['metric_base'] = metric_series(base, metric)
    cand['metric_cand'] = metric_series(cand, metric)

    keep_cols = ['_case_id', 'metric_base', 'prediction']
    keep_cols += [c for c in DISPLAY_COLUMNS if c in base.columns]
    keep_cols += mergeable_columns
    keep_cols = list(dict.fromkeys(keep_cols))
    base = base[[c for c in keep_cols if c in base.columns]].rename(columns={'prediction': 'prediction_base'})

    keep_cols = ['_case_id', 'metric_cand', 'prediction']
    keep_cols += [c for c in DISPLAY_COLUMNS if c in cand.columns]
    keep_cols += mergeable_columns
    keep_cols = list(dict.fromkeys(keep_cols))
    cand = cand[[c for c in keep_cols if c in cand.columns]].rename(columns={'prediction': 'prediction_cand'})

    merged = pd.merge(base, cand, on='_case_id', how='inner', suffixes=('_base', '_cand'))
    coalesce_names = list(dict.fromkeys(DISPLAY_COLUMNS + GROUP_PRIORITY + mergeable_columns))
    merged = coalesce_columns(merged, coalesce_names)

    drop_cols = []
    for name in coalesce_names:
        for suffix in ['_base', '_cand']:
            col = f'{name}{suffix}'
            if col in merged.columns:
                drop_cols.append(col)
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    merged['case_delta'] = (merged['metric_cand'] - merged['metric_base']) * metric.scale
    return merged


def case_direction_mask(df: pd.DataFrame, direction: str) -> pd.Series:
    if direction == 'gain':
        return df['metric_cand'] > df['metric_base']
    if direction == 'regression':
        return df['metric_cand'] < df['metric_base']
    raise ValueError(f'Unknown direction: {direction}')


def case_bucket(base_val: float, cand_val: float, metric: MetricSpec) -> str:
    if pd.isna(base_val) or pd.isna(cand_val):
        return 'invalid'
    if metric.kind == 'binary':
        if cand_val > base_val:
            return 'candidate_fix'
        if cand_val < base_val:
            return 'candidate_drop'
        return 'stable_correct' if cand_val > 0 else 'stable_wrong'
    if cand_val > base_val:
        return 'candidate_gain'
    if cand_val < base_val:
        return 'candidate_drop'
    return 'stable_tie'


def comparison_label(base_val: float, cand_val: float) -> str:
    if pd.isna(base_val) or pd.isna(cand_val):
        return 'invalid'
    if cand_val > base_val:
        return 'candidate_better'
    if cand_val < base_val:
        return 'baseline_better'
    return 'tie'


def annotate_pair_dataframe(df: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame:
    annotated = df.copy()
    annotated['comparison'] = [
        comparison_label(base_val, cand_val)
        for base_val, cand_val in zip(annotated['metric_base'], annotated['metric_cand'])
    ]
    annotated['case_bucket'] = [
        case_bucket(base_val, cand_val, metric)
        for base_val, cand_val in zip(annotated['metric_base'], annotated['metric_cand'])
    ]
    return annotated


def bucket_counts(df: pd.DataFrame) -> dict[str, int]:
    comparison_counts = df['comparison'].value_counts(dropna=False).to_dict()
    case_counts = df['case_bucket'].value_counts(dropna=False).to_dict()
    return {
        'candidate_wins': int(comparison_counts.get('candidate_better', 0)),
        'baseline_wins': int(comparison_counts.get('baseline_better', 0)),
        'ties': int(comparison_counts.get('tie', 0)),
        'stable_correct': int(case_counts.get('stable_correct', 0)) if 'stable_correct' in case_counts else 0,
        'stable_wrong': int(case_counts.get('stable_wrong', 0)) if 'stable_wrong' in case_counts else 0,
    }


def media_hint_from_row(row: pd.Series) -> str:
    for col in ['video_path', 'video_name', 'video', 'image_path']:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return ''


def group_tags_from_row(row: pd.Series, group_columns: list[str]) -> dict[str, Any]:
    tags = {}
    for col in group_columns:
        if col in row and pd.notna(row[col]):
            tags[col] = normalize_scalar(row[col])
    return tags


def case_to_record(
    row: pd.Series,
    dataset: str,
    group_label: str,
    direction: str,
    group_columns: list[str],
) -> dict[str, Any]:
    return {
        'dataset': dataset,
        'group': group_label,
        'direction': direction,
        'case_id': str(row['_case_id']),
        'comparison': row['comparison'],
        'case_bucket': row['case_bucket'],
        'delta': round(float(row['case_delta']), 2),
        'baseline_metric': round(float(row['metric_base']), 4) if pd.notna(row['metric_base']) else None,
        'candidate_metric': round(float(row['metric_cand']), 4) if pd.notna(row['metric_cand']) else None,
        'question': shorten(row.get('question', ''), 240),
        'answer': shorten(row.get('answer', ''), 120),
        'baseline_prediction': shorten(row.get('prediction_base', ''), 180),
        'candidate_prediction': shorten(row.get('prediction_cand', ''), 180),
        'media_hint': media_hint_from_row(row),
        'group_tags': group_tags_from_row(row, group_columns),
    }


def case_inventory_record(
    row: pd.Series,
    dataset: str,
    baseline: str,
    candidate: str,
    metric: MetricSpec,
    group_columns: list[str],
) -> dict[str, Any]:
    tags = group_tags_from_row(row, group_columns)
    record = {
        'dataset': dataset,
        'case_id': str(row['_case_id']),
        'baseline_model': baseline,
        'candidate_model': candidate,
        'metric_key': metric.key,
        'metric_kind': metric.kind,
        'metric_scale': metric.scale,
        'baseline_metric': round(float(row['metric_base']), 6) if pd.notna(row['metric_base']) else None,
        'candidate_metric': round(float(row['metric_cand']), 6) if pd.notna(row['metric_cand']) else None,
        'case_delta': round(float(row['case_delta']), 6) if pd.notna(row['case_delta']) else None,
        'comparison': row['comparison'],
        'case_bucket': row['case_bucket'],
        'media_hint': media_hint_from_row(row),
        'question': shorten(row.get('question', ''), 500),
        'answer': shorten(row.get('answer', ''), 240),
        'baseline_prediction': shorten(row.get('prediction_base', ''), 300),
        'candidate_prediction': shorten(row.get('prediction_cand', ''), 300),
        'group_signature': ' | '.join(f'{k}={v}' for k, v in tags.items()),
        'group_tags': tags,
        'group_tags_json': json.dumps(tags, ensure_ascii=False, sort_keys=True),
    }
    for col in DISPLAY_COLUMNS:
        if col in row and col not in record:
            record[col] = normalize_scalar(row[col])
    for col, value in tags.items():
        record[f'group__{col}'] = value
    return record


def dataset_sort_key(item: DatasetSummary) -> tuple[float, int, str]:
    return (abs(item.delta), item.candidate_wins + item.baseline_wins, item.dataset)


def group_sort_key(item: GroupSummary) -> tuple[float, float, str]:
    return (abs(item.delta), item.swing_rate, item.label)


def plot_dataset_deltas(
    rows: list[DatasetSummary],
    out_path: str,
    top_n: int,
    baseline: str,
    candidate: str,
) -> None:
    if not rows:
        return

    rows = sorted(rows, key=dataset_sort_key, reverse=True)[:top_n]
    rows = sorted(rows, key=lambda x: x.delta)
    labels = [r.dataset for r in rows]
    deltas = [r.delta for r in rows]
    colors = [COLORS['gain'] if d >= 0 else COLORS['regression'] for d in deltas]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.45)))
    y = np.arange(len(rows))
    ax.barh(y, deltas, color=colors, alpha=0.9)
    ax.axvline(0, color='#333333', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(f'{candidate} - {baseline} (points)', fontsize=11)
    ax.set_title('Dataset-Level Delta', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    for yi, delta in zip(y, deltas):
        ha = 'left' if delta >= 0 else 'right'
        offset = 0.3 if delta >= 0 else -0.3
        ax.text(delta + offset, yi, f'{delta:+.2f}', va='center', ha=ha, fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_group_deltas(rows: list[GroupSummary], out_path: str, limit: int) -> None:
    if not rows:
        return

    rows = sorted(rows, key=group_sort_key, reverse=True)[:limit]
    rows = sorted(rows, key=lambda x: x.delta)
    labels = [shorten(r.label, 52) for r in rows]
    deltas = [r.delta for r in rows]
    colors = [COLORS['gain'] if d >= 0 else COLORS['regression'] for d in deltas]

    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.48)))
    y = np.arange(len(rows))
    ax.barh(y, deltas, color=colors, alpha=0.92)
    ax.axvline(0, color='#333333', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Delta (points)', fontsize=11)
    ax.set_title('Top Subgroup Deltas', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    for yi, delta in zip(y, deltas):
        ha = 'left' if delta >= 0 else 'right'
        offset = 0.25 if delta >= 0 else -0.25
        ax.text(delta + offset, yi, f'{delta:+.2f}', va='center', ha=ha, fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_case_balance(
    rows: list[DatasetSummary],
    out_path: str,
    top_n: int,
    baseline: str,
    candidate: str,
) -> None:
    if not rows:
        return

    rows = sorted(rows, key=lambda x: (x.candidate_wins + x.baseline_wins, abs(x.delta), x.dataset), reverse=True)[:top_n]
    rows = sorted(rows, key=lambda x: x.swing_rate)
    labels = [r.dataset for r in rows]
    candidate_rates = [r.candidate_wins / r.sample_count * 100.0 for r in rows]
    baseline_rates = [-(r.baseline_wins / r.sample_count * 100.0) for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.45)))
    y = np.arange(len(rows))
    ax.barh(y, baseline_rates, color=COLORS['regression'], alpha=0.9, label=f'{baseline} only better')
    ax.barh(y, candidate_rates, color=COLORS['gain'], alpha=0.9, label=f'{candidate} only better')
    ax.axvline(0, color='#333333', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Swing cases (% of dataset)', fontsize=11)
    ax.set_title('Case Swing Balance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right', fontsize=8)

    for yi, neg, pos in zip(y, baseline_rates, candidate_rates):
        if neg:
            ax.text(neg - 0.15, yi, f'{neg:.1f}', va='center', ha='right', fontsize=8)
        if pos:
            ax.text(pos + 0.15, yi, f'{pos:.1f}', va='center', ha='left', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def write_report(
    out_path: str,
    baseline: str,
    candidate: str,
    dataset_rows: list[DatasetSummary],
    positive_groups: list[GroupSummary],
    negative_groups: list[GroupSummary],
    cases_by_group: dict[str, dict[str, list[dict[str, Any]]]],
    chart_paths: dict[str, str],
) -> None:
    lines = []
    lines.append(f'# Case Analysis Report: {candidate} vs {baseline}')
    lines.append('')
    lines.append('## Reading Guide')
    lines.append('')
    lines.append('This report follows a three-step flow: dataset delta -> subgroup delta -> representative swing cases.')
    lines.append('The intention is to reduce browsing cost: first see where the score moved, then inspect which ability bucket moved, and only then read a few cases.')
    lines.append('')

    if chart_paths.get('dataset'):
        lines.append('## Dataset Delta')
        lines.append('')
        lines.append(f'![]({osp.basename(chart_paths["dataset"])})')
        lines.append('')

    if chart_paths.get('case_balance'):
        lines.append('## Case Swing Balance')
        lines.append('')
        lines.append(f'![]({osp.basename(chart_paths["case_balance"])})')
        lines.append('')

    if dataset_rows:
        lines.append('| Dataset | Baseline | Candidate | Delta | Candidate Wins | Baseline Wins | Swing Rate | N | Metric |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
        for row in sorted(dataset_rows, key=lambda x: x.delta, reverse=True):
            lines.append(
                f'| {row.dataset} | {row.baseline_score:.2f} | {row.candidate_score:.2f} | '
                f'{row.delta:+.2f} | {row.candidate_wins} | {row.baseline_wins} | '
                f'{row.swing_rate * 100:.1f}% | {row.sample_count} | {row.metric_key} |'
            )
        lines.append('')

    if chart_paths.get('groups'):
        lines.append('## Subgroup Delta')
        lines.append('')
        lines.append(f'![]({osp.basename(chart_paths["groups"])})')
        lines.append('')

    def add_group_section(title: str, groups: list[GroupSummary], direction: str) -> None:
        lines.append(f'## {title}')
        lines.append('')
        if not groups:
            lines.append('No high-signal groups were found.')
            lines.append('')
            return
        for group in groups:
            lines.append(
                f'### {group.label}  '
                f'(N={group.sample_count}, {group.baseline_score:.2f} -> {group.candidate_score:.2f}, '
                f'delta={group.delta:+.2f})'
            )
            lines.append('')
            lines.append(
                f'- Dataset: `{group.dataset}`; dimension: `{group.group_column}`; value: `{group.group_value}`.'
            )
            lines.append(
                f'- Case balance: candidate wins `{group.candidate_wins}`, baseline wins `{group.baseline_wins}`, '
                f'ties `{group.ties}`, swing rate `{group.swing_rate * 100:.1f}%`.'
            )
            if group.stable_correct is not None and group.stable_wrong is not None:
                lines.append(
                    f'- Stable binary buckets: stable correct `{group.stable_correct}`, '
                    f'stable wrong `{group.stable_wrong}`.'
                )
            lines.append('')

            group_cases = cases_by_group.get(group.label, {})
            chosen = group_cases.get(direction, [])
            if not chosen:
                lines.append('No representative cases passed the selection rule.')
                lines.append('')
                continue

            for idx, case in enumerate(chosen, start=1):
                lines.append(f'#### Case {idx}')
                lines.append('')
                lines.append(f'- Case ID: `{case["case_id"]}`')
                if case['media_hint']:
                    lines.append(f'- Media: `{case["media_hint"]}`')
                if case['group_tags']:
                    lines.append(f'- Group tags: `{json.dumps(case["group_tags"], ensure_ascii=False)}`')
                lines.append(f'- Bucket: `{case["case_bucket"]}`; comparison: `{case["comparison"]}`')
                lines.append(f'- Question: {case["question"]}')
                if case['answer']:
                    lines.append(f'- Gold: {case["answer"]}')
                lines.append(
                    f'- Baseline: {case["baseline_prediction"]} '
                    f'(metric={case["baseline_metric"]})'
                )
                lines.append(
                    f'- Candidate: {case["candidate_prediction"]} '
                    f'(metric={case["candidate_metric"]})'
                )
                lines.append(f'- Delta: {case["delta"]:+.2f}')
                lines.append('')

    add_group_section('Top Improvements', positive_groups, 'gain')
    add_group_section('Top Regressions', negative_groups, 'regression')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_ai_prompt(
    out_path: str,
    baseline: str,
    candidate: str,
    dataset_rows: list[DatasetSummary],
    positive_groups: list[GroupSummary],
    negative_groups: list[GroupSummary],
    cases_by_group: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    payload = {
        'baseline': baseline,
        'candidate': candidate,
        'dataset_rows': [asdict(x) for x in dataset_rows],
        'top_improvements': [asdict(x) for x in positive_groups],
        'top_regressions': [asdict(x) for x in negative_groups],
        'cases': cases_by_group,
    }

    lines = []
    lines.append(f'# Prompt For AI Analysis: {candidate} vs {baseline}')
    lines.append('')
    lines.append('You are analyzing ability differences between two video-language models.')
    lines.append('Do not restate raw tables. Focus on: what improved, what regressed, and what type of reasoning or perception shift the changes suggest.')
    lines.append('')
    lines.append('Please produce:')
    lines.append('1. Three concise ability gains with evidence from subgroup deltas and representative cases.')
    lines.append('2. Two regressions or instability patterns.')
    lines.append('3. One paragraph on whether the gains look broad, narrow, or benchmark-specific.')
    lines.append('4. One paragraph on what additional benchmarks or manual checks are still needed.')
    lines.append('5. No claims beyond the provided evidence.')
    lines.append('')
    lines.append('## Structured Payload')
    lines.append('')
    lines.append('```json')
    lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
    lines.append('```')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def flatten_selected_cases(
    positive_groups: list[GroupSummary],
    negative_groups: list[GroupSummary],
    cases_by_group: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    rows = []
    for direction, groups in [('gain', positive_groups), ('regression', negative_groups)]:
        for group in groups:
            for case in cases_by_group.get(group.label, {}).get(direction, []):
                row = dict(case)
                row['selection_group'] = group.label
                row['selection_direction'] = direction
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    base_files = model_score_files(args.work_dir, args.baseline)
    cand_files = model_score_files(args.work_dir, args.candidate)
    common = sorted(set(base_files) & set(cand_files))
    if args.data:
        requested = set(args.data)
        common = [x for x in common if x in requested]

    if not common:
        raise SystemExit('No common row-level `_score` files found for the requested model pair.')

    dataset_rows: list[DatasetSummary] = []
    group_rows: list[GroupSummary] = []
    cases_by_group: dict[str, dict[str, list[dict[str, Any]]]] = {}
    case_inventory: list[dict[str, Any]] = []

    for dataset in common:
        base_df = load_row_level_score(base_files[dataset])
        cand_df = load_row_level_score(cand_files[dataset])
        if base_df is None or cand_df is None:
            continue

        base_spec = infer_metric_spec(base_df)
        cand_spec = infer_metric_spec(cand_df)
        if base_spec is None or cand_spec is None or base_spec.key != cand_spec.key:
            continue

        pair_df = build_pair_dataframe(base_df, cand_df, base_spec)
        if pair_df.empty:
            continue
        pair_df = annotate_pair_dataframe(pair_df, base_spec)

        group_columns = resolve_group_columns(pair_df, args.group_columns)
        counts = bucket_counts(pair_df)
        swing_total = counts['candidate_wins'] + counts['baseline_wins']
        stable_correct = counts['stable_correct'] if base_spec.kind == 'binary' else None
        stable_wrong = counts['stable_wrong'] if base_spec.kind == 'binary' else None

        baseline_score = aggregate_score(pair_df['metric_base'], base_spec)
        candidate_score = aggregate_score(pair_df['metric_cand'], base_spec)
        dataset_rows.append(
            DatasetSummary(
                dataset=dataset,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                delta=candidate_score - baseline_score,
                sample_count=len(pair_df),
                metric_key=base_spec.key,
                candidate_wins=counts['candidate_wins'],
                baseline_wins=counts['baseline_wins'],
                ties=counts['ties'],
                swing_rate=swing_total / len(pair_df) if len(pair_df) else 0.0,
                stable_correct=stable_correct,
                stable_wrong=stable_wrong,
            )
        )

        for _, row in pair_df.iterrows():
            case_inventory.append(
                case_inventory_record(
                    row=row,
                    dataset=dataset,
                    baseline=args.baseline,
                    candidate=args.candidate,
                    metric=base_spec,
                    group_columns=group_columns,
                )
            )

        for col in group_columns:
            sub = pair_df[pair_df[col].notna()].copy()
            if sub.empty:
                continue
            for value, grp in sub.groupby(col):
                if len(grp) < args.min_group_size:
                    continue
                baseline_score = aggregate_score(grp['metric_base'], base_spec)
                candidate_score = aggregate_score(grp['metric_cand'], base_spec)
                if math.isnan(baseline_score) or math.isnan(candidate_score):
                    continue

                grp_counts = bucket_counts(grp)
                grp_swing_total = grp_counts['candidate_wins'] + grp_counts['baseline_wins']
                label = f'{dataset} / {col}={value}'
                group_rows.append(
                    GroupSummary(
                        dataset=dataset,
                        group_column=col,
                        group_value=str(value),
                        label=label,
                        baseline_score=baseline_score,
                        candidate_score=candidate_score,
                        delta=candidate_score - baseline_score,
                        sample_count=len(grp),
                        candidate_wins=grp_counts['candidate_wins'],
                        baseline_wins=grp_counts['baseline_wins'],
                        ties=grp_counts['ties'],
                        swing_rate=grp_swing_total / len(grp) if len(grp) else 0.0,
                        stable_correct=grp_counts['stable_correct'] if base_spec.kind == 'binary' else None,
                        stable_wrong=grp_counts['stable_wrong'] if base_spec.kind == 'binary' else None,
                    )
                )

                cases_by_group.setdefault(label, {})
                for direction in ['gain', 'regression']:
                    direction_df = grp[case_direction_mask(grp, direction)].copy()
                    if direction_df.empty:
                        cases_by_group[label][direction] = []
                        continue
                    direction_df = direction_df.sort_values('case_delta', ascending=(direction != 'gain'))
                    selected = direction_df.head(args.cases_per_group)
                    cases_by_group[label][direction] = [
                        case_to_record(row, dataset, label, direction, group_columns)
                        for _, row in selected.iterrows()
                    ]

    dataset_rows = [x for x in dataset_rows if not math.isnan(x.delta)]
    group_rows = [x for x in group_rows if not math.isnan(x.delta)]
    positive_groups = sorted([x for x in group_rows if x.delta > 0], key=lambda x: (x.delta, x.swing_rate), reverse=True)[
        : args.top_groups
    ]
    negative_groups = sorted([x for x in group_rows if x.delta < 0], key=lambda x: (x.delta, -x.swing_rate))[
        : args.top_groups
    ]

    chart_paths = {}
    if dataset_rows:
        dataset_chart = osp.join(args.out_dir, '01_dataset_delta.png')
        plot_dataset_deltas(dataset_rows, dataset_chart, args.top_datasets, args.baseline, args.candidate)
        chart_paths['dataset'] = dataset_chart

        case_balance_chart = osp.join(args.out_dir, '02_case_balance.png')
        plot_case_balance(dataset_rows, case_balance_chart, args.top_datasets, args.baseline, args.candidate)
        chart_paths['case_balance'] = case_balance_chart

    if group_rows:
        group_chart = osp.join(args.out_dir, '03_group_delta.png')
        plot_group_deltas(group_rows, group_chart, args.top_groups * 2)
        chart_paths['groups'] = group_chart

    report_path = osp.join(args.out_dir, 'report.md')
    write_report(
        out_path=report_path,
        baseline=args.baseline,
        candidate=args.candidate,
        dataset_rows=dataset_rows,
        positive_groups=positive_groups,
        negative_groups=negative_groups,
        cases_by_group=cases_by_group,
        chart_paths=chart_paths,
    )

    ai_prompt_path = osp.join(args.out_dir, 'ai_prompt.md')
    write_ai_prompt(
        out_path=ai_prompt_path,
        baseline=args.baseline,
        candidate=args.candidate,
        dataset_rows=dataset_rows,
        positive_groups=positive_groups,
        negative_groups=negative_groups,
        cases_by_group=cases_by_group,
    )

    dataset_summary_rows = [asdict(x) for x in dataset_rows]
    group_summary_rows = [asdict(x) for x in sorted(group_rows, key=group_sort_key, reverse=True)]
    representative_cases = flatten_selected_cases(positive_groups, negative_groups, cases_by_group)

    dataset_summary_csv = osp.join(args.out_dir, 'dataset_summary.csv')
    group_summary_csv = osp.join(args.out_dir, 'group_summary.csv')
    case_inventory_jsonl = osp.join(args.out_dir, 'case_inventory.jsonl')
    case_inventory_csv = osp.join(args.out_dir, 'case_inventory.csv')
    representative_cases_jsonl = osp.join(args.out_dir, 'representative_cases.jsonl')

    dump_table(dataset_summary_rows, dataset_summary_csv)
    dump_table(group_summary_rows, group_summary_csv)
    dump_jsonl(case_inventory, case_inventory_jsonl)
    dump_table(case_inventory, case_inventory_csv)
    dump_jsonl(representative_cases, representative_cases_jsonl)

    summary = {
        'baseline': args.baseline,
        'candidate': args.candidate,
        'common_datasets': common,
        'dataset_rows': dataset_summary_rows,
        'positive_groups': [asdict(x) for x in positive_groups],
        'negative_groups': [asdict(x) for x in negative_groups],
        'cases_by_group': cases_by_group,
        'chart_paths': chart_paths,
        'report_path': report_path,
        'ai_prompt_path': ai_prompt_path,
        'dataset_summary_csv': dataset_summary_csv,
        'group_summary_csv': group_summary_csv,
        'case_inventory_jsonl': case_inventory_jsonl,
        'case_inventory_csv': case_inventory_csv,
        'representative_cases_jsonl': representative_cases_jsonl,
    }
    dump_json(summary, osp.join(args.out_dir, 'summary.json'))

    print(f'Processed {len(dataset_rows)} datasets with row-level score files.')
    print(f'Report written to: {report_path}')
    print(f'Case inventory written to: {case_inventory_jsonl}')
    print(f'AI prompt written to: {ai_prompt_path}')


if __name__ == '__main__':
    main()
