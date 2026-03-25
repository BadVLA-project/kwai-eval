"""Heatmap visualizations: overall, AoT variants, VideoMME duration."""

import matplotlib.pyplot as plt
import seaborn as sns

from .style import save_fig
from .config import MODEL_LABELS, BASE_MODEL


def plot_master_heatmap(loader, output_dir, formats):
    """Chart 1: Models x all 11 benchmarks overall scores."""
    df = loader.load_overall_matrix()

    fig, ax = plt.subplots(figsize=(14, 7))
    mask = df.isnull()
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', mask=mask,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Score', 'shrink': 0.8},
                annot_kws={'size': 7})

    # Highlight base model row
    base_label = MODEL_LABELS[BASE_MODEL]
    if base_label in df.index:
        idx = list(df.index).index(base_label)
        ax.add_patch(plt.Rectangle((0, idx), df.shape[1], 1,
                                   fill=False, edgecolor='#2c3e50', lw=2))

    ax.set_title('Overall Benchmark Comparison', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    save_fig(fig, 'heatmap_overall', output_dir, formats)


def plot_aot_heatmap(loader, output_dir, formats):
    """Chart 2: Models x 5 AoT sub-benchmarks."""
    from .config import AOT_DATASETS
    df = loader.load_overall_matrix(datasets=AOT_DATASETS)

    fig, ax = plt.subplots(figsize=(10, 7))
    mask = df.isnull()
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlGnBu', mask=mask,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8},
                annot_kws={'size': 8})

    ax.set_title('AoT Benchmark Variants', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)
    fig.tight_layout()
    save_fig(fig, 'heatmap_aot', output_dir, formats)


def plot_videomme_duration_heatmap(loader, output_dir, formats):
    """Chart 3: Models x VideoMME duration (short/medium/long/overall)."""
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
