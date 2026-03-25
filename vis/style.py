"""Matplotlib publication-quality style configuration."""

import matplotlib
matplotlib.use('Agg')  # headless server

import matplotlib.pyplot as plt
import os


def apply_style():
    """Apply clean publication style globally."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def save_fig(fig, name, output_dir, formats=('pdf', 'png')):
    """Save figure in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(output_dir, f'{name}.{fmt}')
        fig.savefig(path, format=fmt)
    plt.close(fig)
    print(f'  Saved: {name} ({", ".join(formats)})')
