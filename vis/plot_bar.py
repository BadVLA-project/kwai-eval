"""Bar chart visualizations: overall ranking, delta vs base."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import save_fig
from .config import MODEL_NAMES, MODEL_LABELS, MODEL_COLORS, BASE_MODEL


def plot_overall_bar(loader, output_dir, formats):
    """Average score per model (horizontal bar, sorted) with per-benchmark dots."""
    df = loader.load_overall_matrix()

    avg = df.mean(axis=1).sort_values(ascending=True)
    colors = []
    for label in avg.index:
        m = next((k for k, v in MODEL_LABELS.items() if v == label), None)
        colors.append(MODEL_COLORS.get(m, '#888888'))

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(avg))
    ax.barh(y, avg, color=colors, edgecolor='white', linewidth=0.5, height=0.6, alpha=0.8)

    # Overlay individual benchmark scores as dots
    for label, yi in zip(avg.index, y):
        vals = df.loc[label].dropna().values
        ax.scatter(vals, np.full_like(vals, yi), color='#333333', s=12, zorder=5, alpha=0.6)

    for yi, val in zip(y, avg):
        if not np.isnan(val):
            ax.text(val + 0.3, yi, f'{val:.1f}', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(avg.index, fontsize=9)
    ax.set_xlabel('Average Score (dots = individual benchmarks)')
    ax.set_title('Model Ranking by Average Score', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, 'bar_overall', output_dir, formats)


def plot_delta_bar(loader, output_dir, formats):
    """Delta vs base model per benchmark — grouped bar chart."""
    df = loader.load_overall_matrix()
    base_label = MODEL_LABELS[BASE_MODEL]

    if base_label not in df.index:
        print('  WARN: base model not found, skipping delta bar')
        return

    base_row = df.loc[base_label]
    ablation_models = [m for m in MODEL_NAMES if m != BASE_MODEL]
    ablation_labels = [MODEL_LABELS[m] for m in ablation_models]

    delta_data = {}
    for m, label in zip(ablation_models, ablation_labels):
        if label in df.index:
            delta_data[label] = df.loc[label] - base_row

    delta_df = pd.DataFrame(delta_data).T

    n_models = len(delta_df.index)
    n_datasets = len(delta_df.columns)
    x = np.arange(n_models)
    width = 0.7 / n_datasets

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, n_datasets))

    for i, col in enumerate(delta_df.columns):
        vals = delta_df[col].values
        ax.bar(x + i * width - 0.35 + width / 2, vals, width,
               label=col, color=cmap[i], edgecolor='white', linewidth=0.3)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(delta_df.index, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Delta Score vs Base')
    ax.set_title('Ablation Improvement over Base Model', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)
    fig.tight_layout()
    save_fig(fig, 'bar_delta', output_dir, formats)
