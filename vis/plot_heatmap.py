"""Heatmap visualizations: overall benchmarks, VideoMME duration."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .style import save_fig
from .config import MODEL_LABELS, BASE_MODEL


def plot_master_heatmap(loader, output_dir, formats):
    """Models x 7 benchmarks — per-column normalized color, actual scores annotated.

    Color encodes relative standing within each benchmark (0=worst, 1=best),
    removing cross-benchmark scale distortion.  Top-3 per column are marked
    with ① ② ③ so the best model per benchmark is instantly visible.
    """
    df = loader.load_overall_matrix()

    # Per-column min-max normalization for color (highlights relative ranking)
    col_min = df.min()
    col_max = df.max()
    col_span = (col_max - col_min).replace(0, 1)
    df_norm = (df - col_min) / col_span  # 0 = worst, 1 = best per benchmark

    # Build custom annotation matrix: score + top-3 rank marker
    _RANK_MARKS = {1: ' ①', 2: ' ②', 3: ' ③'}
    annot = np.full(df.shape, '', dtype=object)
    for j, col in enumerate(df.columns):
        col_ranks = df[col].rank(ascending=False, na_option='bottom')
        for i, idx in enumerate(df.index):
            val = df.loc[idx, col]
            if pd.isna(val):
                annot[i, j] = ''
            else:
                rank = int(col_ranks.loc[idx])
                suffix = _RANK_MARKS.get(rank, '')
                annot[i, j] = f'{val:.1f}{suffix}'

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.55 + 1.5)))
    mask = df.isnull()
    sns.heatmap(df_norm, annot=annot, fmt='', cmap='YlOrRd', mask=mask,
                vmin=0, vmax=1,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Relative score per benchmark (0=worst, 1=best)',
                          'shrink': 0.8},
                annot_kws={'size': 8})

    # Highlight base model row with a bold border
    base_label = MODEL_LABELS[BASE_MODEL]
    if base_label in df.index:
        idx = list(df.index).index(base_label)
        ax.add_patch(plt.Rectangle((0, idx), df.shape[1], 1,
                                   fill=False, edgecolor='#2c3e50', lw=2.5))

    ax.set_title('Overall Benchmark Comparison  (color = per-benchmark rank, ①②③ = top-3)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    save_fig(fig, 'heatmap_overall', output_dir, formats)


def plot_videomme_duration_heatmap(loader, output_dir, formats):
    """Models x VideoMME duration (short/medium/long/overall)."""
    df = loader.load_videomme_duration()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = df.isnull()
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', mask=mask,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8},
                annot_kws={'size': 9})

    ax.set_title('Video-MME Performance by Duration', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.tight_layout()
    save_fig(fig, 'heatmap_vmme_duration', output_dir, formats)
