"""Radar (spider) chart visualizations for ablation comparison."""

import numpy as np
import matplotlib.pyplot as plt

from .style import save_fig
from .config import (
    MODEL_NAMES, MODEL_LABELS, MODEL_COLORS,
    BASE_MODEL, AOT_MODELS, TG_MODELS,
)


def _radar(ax, labels, model_data, title):
    """Draw a radar chart on the given polar axes.

    model_data: list of (label, values, color) tuples
    """
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=7)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    for label, values, color, ls, lw in model_data:
        vals = list(values) + [values[0]]  # close polygon
        ax.plot(angles, vals, color=color, linewidth=lw, linestyle=ls, label=label)
        ax.fill(angles, vals, color=color, alpha=0.05)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)


def plot_aot_radar(loader, output_dir, formats):
    """Chart 4: AoT ablation models (exp1-9) vs base across all benchmarks."""
    df = loader.load_overall_matrix()

    labels = list(df.columns)
    base_label = MODEL_LABELS[BASE_MODEL]

    models_to_plot = []

    # Base model: thick dashed
    if base_label in df.index:
        vals = df.loc[base_label].values
        vals = np.where(np.isnan(vals), 0, vals)
        models_to_plot.append((base_label, vals, '#2c3e50', '--', 2.5))

    # AoT models: solid colored
    for m in AOT_MODELS:
        label = MODEL_LABELS[m]
        if label in df.index:
            vals = df.loc[label].values
            vals = np.where(np.isnan(vals), 0, vals)
            models_to_plot.append((label, vals, MODEL_COLORS[m], '-', 1.2))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    _radar(ax, labels, models_to_plot, 'AoT Ablation: All Benchmarks')
    fig.tight_layout()
    save_fig(fig, 'radar_aot_ablation', output_dir, formats)


def plot_tg_radar(loader, output_dir, formats):
    """Chart 5: TG ablation models vs base across all benchmarks."""
    df = loader.load_overall_matrix()

    labels = list(df.columns)
    base_label = MODEL_LABELS[BASE_MODEL]

    models_to_plot = []

    if base_label in df.index:
        vals = df.loc[base_label].values
        vals = np.where(np.isnan(vals), 0, vals)
        models_to_plot.append((base_label, vals, '#2c3e50', '--', 2.5))

    for m in TG_MODELS:
        label = MODEL_LABELS[m]
        if label in df.index:
            vals = df.loc[label].values
            vals = np.where(np.isnan(vals), 0, vals)
            models_to_plot.append((label, vals, MODEL_COLORS[m], '-', 1.8))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    _radar(ax, labels, models_to_plot, 'TG Ablation: All Benchmarks')
    fig.tight_layout()
    save_fig(fig, 'radar_tg_ablation', output_dir, formats)
