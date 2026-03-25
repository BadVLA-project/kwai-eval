"""Heatmap visualizations: overall benchmarks, VideoMME duration."""

import matplotlib.pyplot as plt
import seaborn as sns

from .style import save_fig
from .config import MODEL_LABELS, BASE_MODEL


def plot_master_heatmap(loader, output_dir, formats):
    """Models x 7 benchmarks (AoTBench averaged) overall scores."""
    df = loader.load_overall_matrix()

    fig, ax = plt.subplots(figsize=(12, 7))
    mask = df.isnull()
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', mask=mask,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Score', 'shrink': 0.8},
                annot_kws={'size': 8})

    # Highlight base model row
    base_label = MODEL_LABELS[BASE_MODEL]
    if base_label in df.index:
        idx = list(df.index).index(base_label)
        ax.add_patch(plt.Rectangle((0, idx), df.shape[1], 1,
                                   fill=False, edgecolor='#2c3e50', lw=2))

    ax.set_title('Overall Benchmark Comparison', fontsize=14, fontweight='bold', pad=12)
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
