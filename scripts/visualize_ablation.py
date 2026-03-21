"""
Comprehensive visualization for ablation evaluation results.

Generates:
  1. Overview heatmap — models × benchmarks score matrix
  2. Radar chart — multi-model comparison across benchmarks
  3. Per-benchmark grouped bars — model comparison
  4. Sub-category breakdown — detailed within-benchmark analysis
     - AoTBench: 5 sub-benchmarks side-by-side
     - MVBench: 20 task types
     - PerceptionTest: area / reasoning dimensions
     - FutureOmni: source-level breakdown
     - CharadesTimeLens: mIoU + R@1 multi-metric
  5. CoT vs non-CoT comparison — diff heatmap + paired bar charts

Usage:
    # Single work_dir
    python scripts/visualize_ablation.py \
        --work-dir /m2v_intern/xuboshen/zgw/VideoProxyMixed/evaluation \
        --out-dir ./eval/aot_ablation/viz

    # CoT vs non-CoT comparison
    python scripts/visualize_ablation.py \\
        --work-dir /path/to/evaluation_no_cot \\
        --work-dir-cot /path/to/evaluation_cot \\
        --out-dir ./eval/aot_ablation/viz
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

# =========================================================================== #
#  Constants
# =========================================================================== #

COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
    '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
]

# Short display names for long model names
MODEL_SHORT_NAMES = {
    'Qwen3-VL-4B-Instruct_aot_ablation_exp1_v2t_binary': 'exp1 (v2t-bin)',
    'Qwen3-VL-4B-Instruct_aot_ablation_exp2_v2t_3way':   'exp2 (v2t-3way)',
    'Qwen3-VL-4B-Instruct_aot_ablation_exp3_t2v_binary': 'exp3 (t2v-bin)',
    'Qwen3-VL-4B-Instruct_aot_ablation_exp4_t2v_3way':   'exp4 (t2v-3way)',
    'Qwen3-VL-4B-Instruct': 'Qwen3-4B (base)',
    'Qwen3-VL-4B-Instruct-mixed-aot': 'mixed-aot',
    'Qwen3-VL-4B-Instruct-seg': 'seg',
}

BENCH_SHORT_NAMES = {
    'AoTBench_ReverseFilm_16frame': 'ReverseFilm',
    'AoTBench_UCF101_16frame': 'UCF101',
    'AoTBench_Rtime_t2v_16frame': 'Rtime-t2v',
    'AoTBench_Rtime_v2t_16frame': 'Rtime-v2t',
    'AoTBench_QA_16frame': 'AoT-QA',
    'FutureOmni_64frame': 'FutureOmni',
    'CharadesTimeLens_1fps': 'TimeLens',
    'MVBench_MP4_1fps': 'MVBench',
    'PerceptionTest_val_16frame': 'PercepTest',
}


def short_model(name: str) -> str:
    return MODEL_SHORT_NAMES.get(name, name)


def short_bench(name: str) -> str:
    return BENCH_SHORT_NAMES.get(name, name)


# =========================================================================== #
#  File I/O helpers
# =========================================================================== #

def _try_load(path: Path):
    try:
        if path.suffix == '.json':
            with open(path) as f:
                return json.load(f)
        if path.suffix == '.csv':
            import pandas as pd
            return pd.read_csv(path)
        if path.suffix == '.xlsx':
            import pandas as pd
            return pd.read_excel(path)
    except Exception:
        return None


def _extract_overall_score(data) -> float | None:
    """Extract main score from a result file (json dict or csv DataFrame)."""
    if isinstance(data, dict):
        return _score_from_json(data)
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return _score_from_csv(data)
    except ImportError:
        pass
    return None


def _score_from_json(data: dict) -> float | None:
    for key in ('accuracy', 'Accuracy', 'acc', 'score', 'mIoU'):
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])
    if 'overall' in data:
        val = data['overall']
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, list) and len(val) >= 3:
            try:
                return float(str(val[2]).rstrip('%'))
            except (ValueError, TypeError):
                pass
        if isinstance(val, str):
            clean = val.rstrip('%')
            try:
                v = float(clean)
                return v * 100 if (not val.endswith('%') and v <= 1.0) else v
            except ValueError:
                pass
    return None


def _score_from_csv(df) -> float | None:
    import pandas as pd
    if df is None or df.empty:
        return None
    if 'category' in df.columns and 'accuracy' in df.columns:
        overall = df[df['category'].str.lower().str.strip() == 'overall']
        if not overall.empty:
            try:
                return float(overall.iloc[0]['accuracy'])
            except (ValueError, TypeError):
                pass
    # PerceptionTest format: split='Overall'
    if 'split' in df.columns and 'accuracy' in df.columns:
        overall = df[df['split'].str.lower().str.strip() == 'overall']
        if not overall.empty:
            try:
                return float(overall.iloc[0]['accuracy'])
            except (ValueError, TypeError):
                pass
    return None


# =========================================================================== #
#  Sub-category extraction per benchmark type
# =========================================================================== #

def _extract_subcategories_csv(df) -> dict[str, float]:
    """Extract {category: accuracy} from standard VLMEvalKit _acc.csv."""
    import pandas as pd
    result = {}
    if df is None or df.empty:
        return result
    if 'category' in df.columns and 'accuracy' in df.columns:
        for _, row in df.iterrows():
            cat = str(row['category']).strip()
            try:
                result[cat] = float(row['accuracy'])
            except (ValueError, TypeError):
                pass
    return result


def _extract_subcategories_perceptiontest(df) -> dict[str, dict[str, float]]:
    """Extract {split_name: {category: accuracy}} from PerceptionTest _acc.csv."""
    import pandas as pd
    result: dict[str, dict[str, float]] = {}
    if df is None or df.empty:
        return result
    if 'split' in df.columns and 'category' in df.columns and 'accuracy' in df.columns:
        for _, row in df.iterrows():
            split = str(row['split']).strip()
            cat = str(row['category']).strip()
            try:
                result.setdefault(split, {})[cat] = float(row['accuracy'])
            except (ValueError, TypeError):
                pass
    return result


def _extract_mvbench_subcategories(data: dict) -> dict[str, float]:
    """Extract {task_type: accuracy} from MVBench _rating.json."""
    result = {}
    for key, val in data.items():
        if isinstance(val, list) and len(val) >= 3:
            try:
                result[key] = float(str(val[2]).rstrip('%'))
            except (ValueError, TypeError):
                pass
        elif isinstance(val, (int, float)):
            result[key] = float(val)
    return result


def _extract_charades_metrics(data: dict) -> dict[str, float]:
    """Extract {metric_name: value} from CharadesTimeLens _score.json."""
    result = {}
    for key, val in data.items():
        if isinstance(val, (int, float)):
            result[key] = float(val)
    return result


# =========================================================================== #
#  Directory scanning — overall + sub-category
# =========================================================================== #

def scan_work_dir(work_dir: str) -> tuple[
    dict[str, dict[str, float]],            # overall: {bench: {model: score}}
    dict[str, dict[str, dict[str, float]]], # subcats: {bench: {model: {subcat: score}}}
]:
    work_dir = Path(work_dir)
    overall: dict[str, dict[str, float]] = defaultdict(dict)
    subcats: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for model_dir in sorted(work_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        candidate_dirs = [model_dir] + sorted(
            [d for d in model_dir.iterdir() if d.is_dir()], reverse=True
        )
        seen: set[str] = set()

        for run_dir in candidate_dirs:
            for fpath in sorted(run_dir.iterdir()):
                if not fpath.is_file():
                    continue
                fname = fpath.name

                # Match result files
                m = re.match(
                    rf'^{re.escape(model_name)}_(.+?)(?:_acc|_score|_eval|_result|_rating)?\.(?:csv|json|xlsx)$',
                    fname,
                )
                if not m:
                    continue
                bench = m.group(1)
                # Deduplicate: prefer the type that gives most info
                file_key = f'{bench}:{fpath.suffix}:{fname}'
                if bench in seen and '_rating' not in fname and '_acc' not in fname and '_score' not in fname:
                    continue

                data = _try_load(fpath)
                if data is None:
                    continue

                # --- Overall score ---
                score = _extract_overall_score(data)
                if score is not None and bench not in seen:
                    overall[bench][model_name] = score
                    seen.add(bench)

                # --- Sub-categories ---
                if '_rating' in fname and isinstance(data, dict):
                    # MVBench rating.json
                    subs = _extract_mvbench_subcategories(data)
                    if subs:
                        subcats[bench][model_name] = subs
                elif '_score' in fname and fpath.suffix == '.json' and isinstance(data, dict):
                    # CharadesTimeLens score.json (multi-metric)
                    metrics = _extract_charades_metrics(data)
                    if metrics:
                        subcats[bench][model_name] = metrics
                elif '_acc' in fname and fpath.suffix == '.csv':
                    import pandas as pd
                    if isinstance(data, pd.DataFrame):
                        if 'split' in data.columns:
                            # PerceptionTest format
                            pt_data = _extract_subcategories_perceptiontest(data)
                            # Flatten: "area: memory" style keys
                            flat = {}
                            for split_name, cats in pt_data.items():
                                if split_name.lower() == 'overall':
                                    continue
                                for cat, acc in cats.items():
                                    if cat.lower() == 'all':
                                        flat[split_name] = acc
                                    else:
                                        flat[f'{split_name}: {cat}'] = acc
                            if flat:
                                subcats[bench][model_name] = flat
                        else:
                            subs = _extract_subcategories_csv(data)
                            # Remove 'overall' from subcats
                            subs.pop('overall', None)
                            subs.pop('Overall', None)
                            if subs:
                                subcats[bench][model_name] = subs

    return overall, subcats


# =========================================================================== #
#  Plot 1: Overview Heatmap
# =========================================================================== #

def plot_heatmap(
    results: dict[str, dict[str, float]],
    out_path: str,
    title: str = 'Model × Benchmark Score Matrix',
):
    """Score matrix heatmap with values annotated."""
    all_models = sorted({m for s in results.values() for m in s})
    benchmarks = sorted(results.keys())
    if not all_models or not benchmarks:
        return

    data = np.full((len(all_models), len(benchmarks)), np.nan)
    for bi, bench in enumerate(benchmarks):
        for mi, model in enumerate(all_models):
            if model in results[bench]:
                data[mi, bi] = results[bench][model]

    fig, ax = plt.subplots(figsize=(max(8, len(benchmarks) * 1.5),
                                    max(4, len(all_models) * 0.8 + 1.5)))

    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        plt.close(fig)
        return
    vmin, vmax = np.nanmin(valid), np.nanmax(valid)
    pad = max(1, (vmax - vmin) * 0.1)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=vmin - pad, vmax=vmax + pad)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

    # Annotate cells
    for mi in range(len(all_models)):
        for bi in range(len(benchmarks)):
            val = data[mi, bi]
            if np.isnan(val):
                ax.text(bi, mi, '—', ha='center', va='center', fontsize=10, color='#999')
            else:
                text_color = 'white' if (val - vmin) / max(vmax - vmin, 1) < 0.3 else 'black'
                ax.text(bi, mi, f'{val:.1f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([short_bench(b) for b in benchmarks], rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels([short_model(m) for m in all_models], fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Score', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


# =========================================================================== #
#  Plot 2: Radar chart
# =========================================================================== #

def plot_radar(results: dict[str, dict[str, float]], out_path: str):
    all_models = sorted({m for s in results.values() for m in s})
    benchmarks = sorted(results.keys())
    num_vars = len(benchmarks)
    if num_vars < 3 or not all_models:
        return

    # Per-axis normalization
    raw = np.full((len(all_models), num_vars), np.nan)
    for bi, bench in enumerate(benchmarks):
        for mi, model in enumerate(all_models):
            if model in results[bench]:
                raw[mi, bi] = results[bench][model]

    normalized = np.full_like(raw, np.nan)
    axis_ranges: dict[str, tuple[float, float]] = {}
    for bi, bench in enumerate(benchmarks):
        col = raw[:, bi]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            axis_ranges[bench] = (0.0, 100.0)
            continue
        lo, hi = float(valid.min()), float(valid.max())
        spread = hi - lo
        pad_val = max(0.5, spread * 0.25)
        axis_min = max(0.0, lo - pad_val)
        axis_max = min(100.0, hi + pad_val)
        rng = max(axis_max - axis_min, 1.0)
        axis_ranges[bench] = (axis_min, axis_max)
        for mi in range(len(all_models)):
            if not np.isnan(raw[mi, bi]):
                normalized[mi, bi] = (raw[mi, bi] - axis_min) / rng * 100

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_deg = np.linspace(0, 360, num_vars, endpoint=False).tolist()
    theta_offset = np.pi / 4

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.set_theta_offset(theta_offset)

    for mi, model in enumerate(all_models):
        vals = normalized[mi]
        if np.all(np.isnan(vals)):
            continue
        loop_vals = np.append(vals, vals[0])
        loop_angles = angles + [angles[0]]
        color = COLORS[mi % len(COLORS)]
        ax.plot(loop_angles, loop_vals, color=color, linewidth=2.2, label=short_model(model))
        if len(all_models) <= 6:
            ax.fill(loop_angles, loop_vals, color=color, alpha=0.12)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([''] * 5)
    ax.tick_params(pad=30)
    ax.set_xticks(angles)
    ax.set_xticklabels([short_bench(b) for b in benchmarks], fontsize=13, fontweight='bold')

    # Overlay axes for per-axis tick labels
    bbox = ax.get_position()
    cx, cy = bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2
    sz = bbox.width / 2
    for i in range(num_vars):
        oa = fig.add_axes([cx - sz, cy - sz, sz * 2, sz * 2],
                          projection='polar', label=f'ov_{i}')
        oa.patch.set_visible(False)
        oa.grid(False)
        oa.xaxis.set_visible(False)
        oa.set_theta_offset(theta_offset)
        axis_min, axis_max = axis_ranges[benchmarks[i]]
        rng = max(axis_max - axis_min, 1.0)
        tick_actual = [axis_min + rng / 5 * k for k in range(2, 6)]
        tick_norm = [(v - axis_min) / rng * 100 for v in tick_actual]
        tick_labels = [f'{v:.1f}' for v in tick_actual]
        oa.set_rgrids(tick_norm, angle=angles_deg[i], labels=tick_labels, fontsize=10)
        oa.spines['polar'].set_visible(False)
        oa.set_ylim(0, 100)

    leg = ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12),
                    fontsize=11, framealpha=0.9, ncol=1)
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    ax.set_title('Model Comparison (per-axis zoom)',
                 fontsize=15, fontweight='bold', pad=35, y=1.08)
    fig.text(0.5, 0.01, 'Each axis independently scaled. Tick labels = actual scores.',
             ha='center', fontsize=10, fontstyle='italic', color='#666')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


# =========================================================================== #
#  Plot 3: Per-benchmark grouped bar
# =========================================================================== #

def plot_benchmark_bar(bench: str, model_scores: dict[str, float], out_path: str):
    models = sorted(model_scores.keys())
    scores = [model_scores[m] for m in models]
    labels = [short_model(m) for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    x = np.arange(len(models))
    bars = ax.bar(x, scores, width=0.6,
                  color=[COLORS[i % len(COLORS)] for i in range(len(models))],
                  edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(short_bench(bench), fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, min(105, max(scores) * 1.15 + 2))
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


# =========================================================================== #
#  Plot 4: Sub-category breakdown (grouped horizontal bars)
# =========================================================================== #

def plot_subcategory_breakdown(
    bench: str,
    model_subcats: dict[str, dict[str, float]],
    out_path: str,
    max_cats: int = 25,
):
    """Horizontal grouped bar chart: sub-categories × models."""
    if not model_subcats:
        return
    models = sorted(model_subcats.keys())
    # Collect all categories across models
    all_cats = sorted({c for subs in model_subcats.values() for c in subs if c.lower() != 'overall'})
    if not all_cats:
        return
    if len(all_cats) > max_cats:
        # Keep top N by average score
        cat_avg = {}
        for c in all_cats:
            vals = [model_subcats[m].get(c, 0) for m in models]
            cat_avg[c] = np.mean(vals)
        all_cats = sorted(cat_avg, key=cat_avg.get, reverse=True)[:max_cats]

    n_cats = len(all_cats)
    n_models = len(models)
    bar_h = 0.7 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(10, 8), max(6, n_cats * 0.45 + 2)))
    y = np.arange(n_cats)

    for mi, model in enumerate(models):
        offset = (mi - n_models / 2 + 0.5) * bar_h
        vals = [model_subcats[model].get(c, 0) for c in all_cats]
        ax.barh(y + offset, vals, height=bar_h * 0.9,
                label=short_model(model),
                color=COLORS[mi % len(COLORS)],
                edgecolor='white', linewidth=0.5)
        # Value labels
        for i, val in enumerate(vals):
            if val > 0:
                ax.text(val + 0.3, y[i] + offset, f'{val:.1f}',
                        va='center', fontsize=7, color='#333')

    ax.set_yticks(y)
    ax.set_yticklabels(all_cats, fontsize=9)
    ax.set_xlabel('Score', fontsize=11)
    ax.set_title(f'{short_bench(bench)} — Sub-category Breakdown',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.8)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


# =========================================================================== #
#  Plot 5: AoTBench detail — 5 sub-benchmarks as grouped bars
# =========================================================================== #

def plot_aotbench_detail(
    results: dict[str, dict[str, float]],
    out_path: str,
):
    """AoTBench 5 sub-benchmarks: grouped bar with each sub as a cluster."""
    aot_benches = sorted([b for b in results if b.startswith('AoTBench')])
    if len(aot_benches) < 2:
        return
    all_models = sorted({m for b in aot_benches for m in results[b]})
    if not all_models:
        return

    n_bench = len(aot_benches)
    n_model = len(all_models)
    width = 0.75 / max(n_model, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_bench * 2.5), 6))
    x = np.arange(n_bench)

    for mi, model in enumerate(all_models):
        offset = (mi - n_model / 2 + 0.5) * width
        vals = [results[b].get(model, 0) for b in aot_benches]
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=short_model(model),
                      color=COLORS[mi % len(COLORS)],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([short_bench(b) for b in aot_benches], rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('AoTBench — Sub-benchmark Comparison', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.8, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # Add average line per model
    for mi, model in enumerate(all_models):
        vals = [results[b].get(model, np.nan) for b in aot_benches]
        avg = np.nanmean(vals)
        ax.axhline(y=avg, color=COLORS[mi % len(COLORS)], linestyle=':', alpha=0.4, linewidth=1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


# =========================================================================== #
#  Plot 6: CoT vs non-CoT comparison
# =========================================================================== #

def plot_cot_comparison(
    results_base: dict[str, dict[str, float]],
    results_cot: dict[str, dict[str, float]],
    out_path: str,
):
    """Diff heatmap: CoT score - base score per model × benchmark."""
    all_models = sorted(
        {m for r in (results_base, results_cot) for s in r.values() for m in s}
    )
    all_benches = sorted(
        {b for r in (results_base, results_cot) for b in r}
    )
    if not all_models or not all_benches:
        return

    base_arr = np.full((len(all_models), len(all_benches)), np.nan)
    cot_arr = np.full_like(base_arr, np.nan)
    for bi, bench in enumerate(all_benches):
        for mi, model in enumerate(all_models):
            if model in results_base.get(bench, {}):
                base_arr[mi, bi] = results_base[bench][model]
            if model in results_cot.get(bench, {}):
                cot_arr[mi, bi] = results_cot[bench][model]

    diff = cot_arr - base_arr

    fig, axes = plt.subplots(1, 3, figsize=(max(20, len(all_benches) * 4), max(4, len(all_models) * 0.9 + 2)),
                             gridspec_kw={'width_ratios': [1, 1, 1]})

    labels_m = [short_model(m) for m in all_models]
    labels_b = [short_bench(b) for b in all_benches]

    # --- Base scores ---
    valid_all = np.concatenate([base_arr[~np.isnan(base_arr)], cot_arr[~np.isnan(cot_arr)]])
    if len(valid_all) == 0:
        plt.close(fig)
        return
    vmin, vmax = np.nanmin(valid_all), np.nanmax(valid_all)
    pad = max(1, (vmax - vmin) * 0.05)
    norm_score = mcolors.Normalize(vmin=vmin - pad, vmax=vmax + pad)

    for ax_idx, (arr, title) in enumerate([(base_arr, 'No CoT'), (cot_arr, 'With CoT')]):
        ax = axes[ax_idx]
        im = ax.imshow(arr, cmap='RdYlGn', norm=norm_score, aspect='auto')
        for mi in range(len(all_models)):
            for bi in range(len(all_benches)):
                val = arr[mi, bi]
                if np.isnan(val):
                    ax.text(bi, mi, '—', ha='center', va='center', fontsize=9, color='#999')
                else:
                    ax.text(bi, mi, f'{val:.1f}', ha='center', va='center',
                            fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(all_benches)))
        ax.set_xticklabels(labels_b, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(all_models)))
        ax.set_yticklabels(labels_m if ax_idx == 0 else [''] * len(all_models), fontsize=9)
        ax.set_title(title, fontsize=13, fontweight='bold')

    # --- Diff heatmap ---
    ax = axes[2]
    valid_diff = diff[~np.isnan(diff)]
    if len(valid_diff) == 0:
        plt.close(fig)
        return
    abs_max = max(abs(np.nanmin(valid_diff)), abs(np.nanmax(valid_diff)), 0.5)
    norm_diff = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im_diff = ax.imshow(diff, cmap='RdBu', norm=norm_diff, aspect='auto')
    for mi in range(len(all_models)):
        for bi in range(len(all_benches)):
            val = diff[mi, bi]
            if np.isnan(val):
                ax.text(bi, mi, '—', ha='center', va='center', fontsize=9, color='#999')
            else:
                sign = '+' if val > 0 else ''
                color = '#1a7a2e' if val > 0 else '#c0392b' if val < 0 else '#333'
                ax.text(bi, mi, f'{sign}{val:.1f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)
    ax.set_xticks(range(len(all_benches)))
    ax.set_xticklabels(labels_b, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels([''] * len(all_models), fontsize=9)
    ax.set_title('Δ (CoT − Base)', fontsize=13, fontweight='bold')
    fig.colorbar(im_diff, ax=ax, shrink=0.7, aspect=25, pad=0.03, label='Score Diff')

    fig.suptitle('CoT vs Non-CoT Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


def plot_cot_paired_bars(
    results_base: dict[str, dict[str, float]],
    results_cot: dict[str, dict[str, float]],
    out_path: str,
):
    """Paired bar chart: for each benchmark, show base vs CoT per model."""
    all_benches = sorted({b for r in (results_base, results_cot) for b in r})
    all_models = sorted({m for r in (results_base, results_cot) for s in r.values() for m in s})
    if not all_models or not all_benches:
        return

    n_cols = min(3, len(all_benches))
    n_rows = (len(all_benches) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, bench in enumerate(all_benches):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        models_here = [m for m in all_models
                       if m in results_base.get(bench, {}) or m in results_cot.get(bench, {})]
        if not models_here:
            ax.set_visible(False)
            continue

        x = np.arange(len(models_here))
        w = 0.35
        base_vals = [results_base.get(bench, {}).get(m, 0) for m in models_here]
        cot_vals = [results_cot.get(bench, {}).get(m, 0) for m in models_here]

        ax.bar(x - w / 2, base_vals, w, label='No CoT', color='#4C72B0', edgecolor='white')
        ax.bar(x + w / 2, cot_vals, w, label='CoT', color='#DD8452', edgecolor='white')

        # Diff annotations
        for i, (bv, cv) in enumerate(zip(base_vals, cot_vals)):
            if bv > 0 or cv > 0:
                diff = cv - bv
                sign = '+' if diff > 0 else ''
                color = '#1a7a2e' if diff > 0 else '#c0392b'
                ax.text(x[i], max(bv, cv) + 0.5, f'{sign}{diff:.1f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([short_model(m) for m in models_here], rotation=20, ha='right', fontsize=8)
        ax.set_title(short_bench(bench), fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(len(all_benches), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle('CoT vs Non-CoT — Per Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


# =========================================================================== #
#  Merged overview (AoTBench average + others)
# =========================================================================== #

def merge_aotbench(results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Merge AoTBench_* sub-benchmarks into a single 'AoTBench (avg)' entry."""
    merged = {}
    aot_benches = []
    for bench in sorted(results):
        if bench.startswith('AoTBench'):
            aot_benches.append(bench)
        else:
            merged[bench] = results[bench]

    if aot_benches:
        all_models = sorted({m for b in aot_benches for m in results[b]})
        avg_scores = {}
        for model in all_models:
            scores = [results[b][model] for b in aot_benches if model in results[b]]
            if scores:
                avg_scores[model] = sum(scores) / len(scores)
        merged['AoTBench (avg)'] = avg_scores
    return merged


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive visualization for ablation eval results',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--work-dir', required=True,
                        help='VLMEvalKit output directory (non-CoT or main)')
    parser.add_argument('--work-dir-cot', default=None,
                        help='VLMEvalKit output directory for CoT results (optional)')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save plots (default: work_dir/plots)')
    args = parser.parse_args()

    out_dir = args.out_dir or osp.join(args.work_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # --- Scan base results ---
    print(f'Scanning base: {args.work_dir}')
    overall, subcats = scan_work_dir(args.work_dir)
    if not overall:
        print('No result files found.'); sys.exit(1)

    all_models = sorted({m for s in overall.values() for m in s})
    print(f'Found {len(overall)} benchmark(s), {len(all_models)} model(s)\n')

    # --- 1. AoTBench detail (before merging) ---
    print('=' * 60)
    print('[AoTBench Detail]')
    aot_keys = [b for b in overall if b.startswith('AoTBench')]
    if len(aot_keys) >= 2:
        plot_aotbench_detail(overall, osp.join(out_dir, '01_aotbench_detail.png'))

    # --- 2. Per-benchmark overall bars ---
    print('\n[Per-Benchmark Bars]')
    for bench, model_scores in sorted(overall.items()):
        safe = re.sub(r'[^\w\-]', '_', bench)
        plot_benchmark_bar(bench, model_scores, osp.join(out_dir, f'02_bar_{safe}.png'))

    # --- 3. Sub-category breakdowns ---
    print('\n[Sub-category Breakdowns]')
    for bench, model_subs in sorted(subcats.items()):
        if model_subs:
            safe = re.sub(r'[^\w\-]', '_', bench)
            plot_subcategory_breakdown(bench, model_subs, osp.join(out_dir, f'03_subcat_{safe}.png'))

    # --- 4. Merged overview (AoTBench avg + others) ---
    print('\n[Overview]')
    merged = merge_aotbench(overall)
    plot_heatmap(merged, osp.join(out_dir, '04_heatmap_overview.png'))

    # --- 5. Radar ---
    print('\n[Radar]')
    plot_radar(merged, osp.join(out_dir, '05_radar.png'))

    # --- 6. Full heatmap (all sub-benchmarks) ---
    plot_heatmap(overall, osp.join(out_dir, '06_heatmap_full.png'),
                 title='All Sub-benchmarks × Models')

    # --- 7. CoT comparison (if provided) ---
    if args.work_dir_cot:
        print(f'\nScanning CoT: {args.work_dir_cot}')
        overall_cot, subcats_cot = scan_work_dir(args.work_dir_cot)
        if overall_cot:
            merged_cot = merge_aotbench(overall_cot)
            print('\n[CoT Comparison]')
            plot_cot_comparison(merged, merged_cot, osp.join(out_dir, '07_cot_heatmap.png'))
            plot_cot_paired_bars(merged, merged_cot, osp.join(out_dir, '08_cot_paired_bars.png'))

            # CoT sub-category comparison for benchmarks that have subcats
            for bench in sorted(set(subcats) & set(subcats_cot)):
                if subcats[bench] and subcats_cot[bench]:
                    # Merge base + cot into one plot by renaming models
                    combined = {}
                    for m, subs in subcats[bench].items():
                        combined[f'{short_model(m)} (base)'] = subs
                    for m, subs in subcats_cot[bench].items():
                        combined[f'{short_model(m)} (CoT)'] = subs
                    safe = re.sub(r'[^\w\-]', '_', bench)
                    plot_subcategory_breakdown(
                        bench, combined,
                        osp.join(out_dir, f'09_cot_subcat_{safe}.png'),
                    )

    # --- Summary table to console ---
    print('\n' + '=' * 60)
    print('SCORE SUMMARY')
    print('=' * 60)
    benches_sorted = sorted(overall.keys())
    header = f'{"Model":<25s}' + ''.join(f'{short_bench(b):>12s}' for b in benches_sorted)
    print(header)
    print('-' * len(header))
    for model in all_models:
        row = f'{short_model(model):<25s}'
        for bench in benches_sorted:
            val = overall[bench].get(model)
            row += f'{val:>12.1f}' if val is not None else f'{"—":>12s}'
        print(row)
    print()

    print(f'Done. All plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
