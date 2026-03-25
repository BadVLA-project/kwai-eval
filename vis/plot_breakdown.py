"""Sub-category breakdown charts for detailed analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .style import save_fig
from .config import MODEL_LABELS, MODEL_COLORS, MODEL_NAMES, BASE_MODEL


def _plot_delta_heatmap(df, title, fig_id, output_dir, formats,
                        figsize=None, annot_size=8):
    """Plot a sub-task delta heatmap (rows=ablation models, cols=sub-tasks, cell=Δ vs base).

    Green = improvement over base, Red = regression.
    The base model row is excluded from the heatmap body; its scores are the reference.
    """
    base_label = MODEL_LABELS[BASE_MODEL]
    if base_label not in df.index:
        print(f'  WARN: base model not found, skipping delta heatmap for {fig_id}')
        return

    base_row = df.loc[base_label]
    delta_df = df.drop(index=base_label).subtract(base_row, axis=1).dropna(how='all')

    if delta_df.empty:
        return

    valid = delta_df.values[~pd.isnull(delta_df.values)]
    vmax = float(max(abs(valid).max() if len(valid) else 1.0, 0.5))

    auto_w = max(10, len(delta_df.columns) * 0.75 + 2)
    auto_h = max(3.5, len(delta_df) * 0.6 + 1.5)
    if figsize is None:
        figsize = (auto_w, auto_h)
    else:
        figsize = (figsize[0] or auto_w, figsize[1] or auto_h)

    fig, ax = plt.subplots(figsize=figsize)
    mask = delta_df.isnull()
    sns.heatmap(delta_df, annot=True, fmt='+.1f', cmap='RdYlGn', mask=mask,
                center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': f'Δ vs {base_label}', 'shrink': 0.7},
                annot_kws={'size': annot_size})

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    save_fig(fig, fig_id, output_dir, formats)


def plot_mvbench_breakdown(loader, output_dir, formats):
    """Chart 9: MVBench 20 sub-tasks heatmap."""
    df = loader.load_mvbench_tasks()
    if df.empty:
        print('  WARN: no MVBench data, skipping breakdown')
        return

    # Sort columns by mean performance for readability
    col_order = df.mean().sort_values(ascending=False).index
    df = df[col_order]

    fig, ax = plt.subplots(figsize=(16, 7))
    mask = df.isnull()
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', mask=mask,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.6},
                annot_kws={'size': 6})

    ax.set_title('MVBench Sub-task Breakdown', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'breakdown_mvbench', output_dir, formats)

    _plot_delta_heatmap(df, 'MVBench Sub-task Delta vs Base',
                        'delta_mvbench', output_dir, formats,
                        figsize=(18, None), annot_size=7)


def plot_videomme_tasktype(loader, output_dir, formats):
    """Chart 10: VideoMME 12 task types heatmap."""
    df = loader.load_videomme_tasktype()
    if df.empty:
        print('  WARN: no VideoMME task_type data, skipping breakdown')
        return

    col_order = df.mean().sort_values(ascending=False).index
    df = df[col_order]

    fig, ax = plt.subplots(figsize=(16, 7))
    mask = df.isnull()
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', mask=mask,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.6},
                annot_kws={'size': 6})

    ax.set_title('Video-MME Task Type Breakdown', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'breakdown_vmme_task', output_dir, formats)

    _plot_delta_heatmap(df, 'Video-MME Task Type Delta vs Base',
                        'delta_vmme_task', output_dir, formats)


def plot_videoholmes_breakdown(loader, output_dir, formats):
    """Chart 11: Video_Holmes 7 question types grouped bar."""
    df = loader.load_videoholmes_types()
    if df.empty:
        print('  WARN: no Video_Holmes data, skipping breakdown')
        return

    n_models = len(df.index)
    n_types = len(df.columns)
    x = np.arange(n_types)
    width = 0.8 / n_models

    # Map model labels to colors
    label_to_color = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in MODEL_NAMES}

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_label in enumerate(df.index):
        vals = df.loc[model_label].values
        color = label_to_color.get(model_label, '#888888')
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=model_label, color=color, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=0, fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Video-Holmes Question Type Breakdown', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)
    fig.tight_layout()
    save_fig(fig, 'breakdown_videoholmes', output_dir, formats)

    _plot_delta_heatmap(df, 'Video-Holmes Question Type Delta vs Base',
                        'delta_videoholmes', output_dir, formats)


def plot_perceptiontest_breakdown(loader, output_dir, formats):
    """Chart 12: PerceptionTest 3. dimensions (area, reasoning, tag) multi-panel."""
    dims = loader.load_perception_dims()
    if not dims:
        print('  WARN: no PerceptionTest data, skipping breakdown')
        return

    dim_names = sorted(dims.keys())
    n_dims = len(dim_names)

    fig, axes = plt.subplots(1, n_dims, figsize=(6 * n_dims, 6), sharey=True)
    if n_dims == 1:
        axes = [axes]

    label_to_color = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in MODEL_NAMES}

    for ax, dim in zip(axes, dim_names):
        df = dims[dim]
        if df.empty:
            continue

        n_models = len(df.index)
        n_cats = len(df.columns)
        x = np.arange(n_cats)
        width = 0.8 / n_models

        for i, model_label in enumerate(df.index):
            vals = df.loc[model_label].values
            color = label_to_color.get(model_label, '#888888')
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=model_label, color=color, edgecolor='white', linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(df.columns, rotation=35, ha='right', fontsize=7)
        ax.set_title(f'PerceptionTest: {dim}', fontsize=11, fontweight='bold')
        if dim == dim_names[0]:
            ax.set_ylabel('Accuracy (%)')

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), loc='upper center',
               fontsize=7, ncol=min(6, len(labels)))
    fig.suptitle('PerceptionTest Multi-Dimension Breakdown', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'breakdown_perceptiontest', output_dir, formats)

    # Delta heatmap per split
    for dim_name, df_dim in zip(dim_names, [dims[d] for d in dim_names]):
        _plot_delta_heatmap(df_dim,
                            f'PerceptionTest {dim_name} Delta vs Base',
                            f'delta_perceptiontest_{dim_name.lower()}',
                            output_dir, formats)


def plot_charades_breakdown(loader, output_dir, formats):
    """Chart 13: CharadesTimeLens 4 metrics (mIoU, R@1_IoU=0.3/0.5/0.7) grouped bar."""
    df = loader.load_charades_metrics()
    if df.empty:
        print('  WARN: no CharadesTimeLens data, skipping breakdown')
        return

    label_to_color = {MODEL_LABELS[m]: MODEL_COLORS[m] for m in MODEL_NAMES}

    n_models = len(df.index)
    n_metrics = len(df.columns)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_label in enumerate(df.index):
        vals = df.loc[model_label].values
        color = label_to_color.get(model_label, '#888888')
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=model_label, color=color, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=0, fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_title('CharadesTimeLens Metrics Breakdown', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)
    fig.tight_layout()
    save_fig(fig, 'breakdown_charades', output_dir, formats)

    _plot_delta_heatmap(df, 'CharadesTimeLens Metrics Delta vs Base',
                        'delta_charades', output_dir, formats, figsize=(8, None))
