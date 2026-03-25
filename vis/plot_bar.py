"""Bar chart visualizations: overall ranking, delta heatmap, rank bump."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Delta vs base — diverging heatmap (rows=models, cols=benchmarks).

    Green = improvement over base, Red = regression.
    Replaces the old grouped bar chart which became unreadable with many models.
    """
    df = loader.load_overall_matrix()
    base_label = MODEL_LABELS[BASE_MODEL]

    if base_label not in df.index:
        print('  WARN: base model not found, skipping delta heatmap')
        return

    base_row = df.loc[base_label]
    delta_df = df.drop(index=base_label).subtract(base_row, axis=1)

    if delta_df.empty:
        print('  WARN: no ablation models for delta heatmap')
        return

    valid = delta_df.values[~pd.isnull(delta_df.values)]
    vmax = float(max(abs(valid).max() if len(valid) else 1.0, 1.0))

    h = max(3.5, len(delta_df) * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(10, h))
    mask = delta_df.isnull()
    sns.heatmap(delta_df, annot=True, fmt='+.1f', cmap='RdYlGn', mask=mask,
                center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': f'Δ vs {base_label}', 'shrink': 0.8},
                annot_kws={'size': 9, 'fontweight': 'bold'})

    ax.set_title(f'Ablation Delta vs Base  ({base_label})',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    save_fig(fig, 'heatmap_delta', output_dir, formats)


def plot_rank_bump(loader, output_dir, formats):
    """Cross-benchmark rank bump chart.

    x = benchmark, y = rank (1=best at top), one line per model.
    Immediately reveals which model is consistently strong vs "偏科".
    """
    df = loader.load_overall_matrix()
    if df.empty:
        print('  WARN: no data for rank bump chart')
        return

    # Rank per column: 1 = highest score
    ranks = df.rank(ascending=False, na_option='bottom')

    benchmarks = list(ranks.columns)
    n_bench = len(benchmarks)
    n_models = len(ranks.index)

    fig, ax = plt.subplots(figsize=(max(8, n_bench * 1.6), max(5, n_models * 0.5 + 2)))

    label_to_color = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in MODEL_NAMES}
    base_label = MODEL_LABELS[BASE_MODEL]

    x = np.arange(n_bench)
    for model_label in ranks.index:
        y = ranks.loc[model_label].values
        color = label_to_color.get(model_label, '#888888')
        lw = 2.5 if model_label == base_label else 1.2
        ls = '--' if model_label == base_label else '-'
        alpha = 1.0 if model_label == base_label else 0.75
        ax.plot(x, y, marker='o', markersize=6, color=color,
                linewidth=lw, linestyle=ls, alpha=alpha, label=model_label)
        # Annotate rank at each benchmark
        for xi, yi in zip(x, y):
            if not np.isnan(yi):
                ax.text(xi, yi - 0.18, str(int(yi)), ha='center', va='bottom',
                        fontsize=6, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(1, n_models + 1))
    ax.set_yticklabels([f'#{i}' for i in range(1, n_models + 1)], fontsize=8)
    ax.invert_yaxis()  # rank 1 at top
    ax.set_ylabel('Rank (1 = best per benchmark)')
    ax.set_title('Cross-Benchmark Ranking', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.25, linestyle=':')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)
    fig.tight_layout()
    save_fig(fig, 'bump_rank', output_dir, formats)
