"""Radar (spider) chart visualizations for ablation comparison.

Per-axis min-max normalization so differences between models are visible
even when benchmarks have very different score ranges.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import save_fig
from .config import (
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS,
    BASE_MODEL, AOT_MODELS, TG_MODELS,
)


def _normalize_per_axis(df):
    """Min-max normalize each column (benchmark) to [0, 1].

    Adds a small margin so no model sits exactly at the edge.
    Returns (normalized_df, axis_ranges) for tick labeling.
    """
    col_min = df.min()
    col_max = df.max()
    span = col_max - col_min
    span = span.replace(0, 1)  # avoid div-by-zero for constant columns
    # Expand range by 10% on each side
    margin = span * 0.1
    low = col_min - margin
    high = col_max + margin
    normed = (df - low) / (high - low)
    ranges = {col: (low[col], high[col]) for col in df.columns}
    return normed, ranges


def _radar(ax, labels, model_data, title, axis_ranges=None):
    """Draw a radar chart on the given polar axes.

    model_data: list of (label, values, color, linestyle, linewidth)
    axis_ranges: optional dict {label: (low, high)} for annotating actual score range
    """
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)

    # Axis labels with score range annotation
    if axis_ranges:
        tick_labels = []
        for lbl in labels:
            lo, hi = axis_ranges.get(lbl, (0, 100))
            tick_labels.append(f'{lbl}\n({lo:.0f}-{hi:.0f})')
    else:
        tick_labels = labels

    ax.set_thetagrids(np.degrees(angles[:-1]), tick_labels, fontsize=7)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=25)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])  # hide radial ticks since they're normalized

    for label, values, color, ls, lw in model_data:
        vals = list(values) + [values[0]]
        ax.plot(angles, vals, color=color, linewidth=lw, linestyle=ls, label=label)
        ax.fill(angles, vals, color=color, alpha=0.05)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7)


_RADAR_MAX_MODELS = 5  # max ablation models on a single radar (+ base)


def _prepare_radar_data(df, models_to_include, base_label):
    """Normalize df and build plot data list.

    Only the top-_RADAR_MAX_MODELS ablation models (by mean score) are kept
    to prevent the radar from becoming an unreadable tangle of overlapping lines.
    """
    # Filter to only models present in df
    present = [m for m in models_to_include if MODEL_LABELS[m] in df.index]
    if base_label not in df.index:
        return [], [], {}, []

    # Keep top-N by mean score (avoids clutter when many ablations exist)
    if len(present) > _RADAR_MAX_MODELS:
        scored = sorted(present,
                        key=lambda m: df.loc[MODEL_LABELS[m]].mean(skipna=True),
                        reverse=True)
        present = scored[:_RADAR_MAX_MODELS]

    # Subset to relevant rows for normalization
    relevant_labels = [base_label] + [MODEL_LABELS[m] for m in present]
    relevant_labels = [lbl for lbl in relevant_labels if lbl in df.index]
    sub = df.loc[relevant_labels].copy()
    sub = sub.fillna(sub.mean())  # fill NaN with column mean for normalization

    normed, ranges = _normalize_per_axis(sub)
    return normed, list(df.columns), ranges, present


def plot_aot_radar(loader, output_dir, formats):
    """Chart 4: AoT ablation models (exp1-9) vs base, per-axis normalized."""
    df = loader.load_overall_matrix()
    base_label = MODEL_LABELS[BASE_MODEL]
    normed, labels, ranges, present = _prepare_radar_data(df, AOT_MODELS, base_label)

    if len(normed) == 0:
        print('  WARN: insufficient data for AoT radar')
        return

    models_to_plot = []
    if base_label in normed.index:
        vals = normed.loc[base_label].values
        models_to_plot.append((base_label, vals, '#2c3e50', '--', 2.5))

    for m in present:
        label = MODEL_LABELS[m]
        if label in normed.index:
            vals = normed.loc[label].values
            models_to_plot.append((label, vals, MODEL_COLORS[m], '-', 1.2))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    _radar(ax, labels, models_to_plot, 'AoT Ablation (normalized per-axis)', ranges)
    fig.tight_layout()
    save_fig(fig, 'radar_aot_ablation', output_dir, formats)


def plot_tg_radar(loader, output_dir, formats):
    """Chart 5: TG ablation models vs base, per-axis normalized."""
    df = loader.load_overall_matrix()
    base_label = MODEL_LABELS[BASE_MODEL]
    normed, labels, ranges, present = _prepare_radar_data(df, TG_MODELS, base_label)

    if len(normed) == 0:
        print('  WARN: insufficient data for TG radar')
        return

    models_to_plot = []
    if base_label in normed.index:
        vals = normed.loc[base_label].values
        models_to_plot.append((base_label, vals, '#2c3e50', '--', 2.5))

    for m in present:
        label = MODEL_LABELS[m]
        if label in normed.index:
            vals = normed.loc[label].values
            models_to_plot.append((label, vals, MODEL_COLORS[m], '-', 1.8))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    _radar(ax, labels, models_to_plot, 'TG Ablation (normalized per-axis)', ranges)
    fig.tight_layout()
    save_fig(fig, 'radar_tg_ablation', output_dir, formats)


def plot_aot_subsets_radar(loader, output_dir, formats):
    """Radar chart: AoT 5 sub-benchmarks as axes, each model as a polygon."""
    df = loader.load_aot_subsets()
    if df is None or df.empty:
        print('  WARN: no AoT subset data found')
        return

    # include base + all aot ablation models
    all_models = [BASE_MODEL] + AOT_MODELS
    present_labels = [MODEL_LABELS[m] for m in all_models if MODEL_LABELS[m] in df.index]
    if not present_labels:
        print('  WARN: no models found for AoT subset radar')
        return

    sub = df.loc[present_labels].copy()
    sub = sub.fillna(sub.mean())
    normed, ranges = _normalize_per_axis(sub)
    labels = list(df.columns)

    models_to_plot = []
    base_label = MODEL_LABELS[BASE_MODEL]
    if base_label in normed.index:
        models_to_plot.append((base_label, normed.loc[base_label].values, '#2c3e50', '--', 2.5))
    for m in AOT_MODELS:
        lbl = MODEL_LABELS[m]
        if lbl in normed.index:
            models_to_plot.append((lbl, normed.loc[lbl].values, MODEL_COLORS[m], '-', 1.2))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    _radar(ax, labels, models_to_plot, 'AoT Sub-benchmarks (normalized per-axis)', ranges)
    fig.tight_layout()
    save_fig(fig, 'radar_aot_subsets', output_dir, formats)
