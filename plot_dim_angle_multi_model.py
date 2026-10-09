# Multi-panel figure: one panel per (dataset, model), each showing the
# split-half DiM angle vs. subsample size n, with the fitted tan^2(theta) =
# C/n curve (see fit_dim_angle_c.py) overlaid and flagged if the fit is
# poor (R^2 < 0.9), across TruthfulQA / Refusal / Sycophancy.

import numpy as np
import pandas as pd
# Palette: a normal (good 1/n fit) panel in blue, a flagged (poor fit /
# heterogeneous) panel in orange -- status-like use, not a categorical
# series, so only two colors appear and each is labeled directly in its
# panel title rather than via a legend.
from common import setup_paper_style, COLOR_OK, COLOR_FLAG, fit_tan2_c
setup_paper_style()
import matplotlib.pyplot as plt

# All panels use split-half angle: the angle between two DISJOINT n-sample
# DiM estimates drawn from the same pool. This has no degeneracy as n
# approaches the pool size (unlike comparing a subsample to the full-data
# direction, which collapses to ~0 once the subsample basically *is* the
# full pool -- see the old version of this script / earlier figure).
panels = [
    ('TruthfulQA', 'llama3-8B-Instruct', 'results/dim_angle_vs_n_llama3_8B_instruct_splithalf.csv'),
    ('TruthfulQA', 'qwen2.5-7B-Instruct', 'results/dim_angle_vs_n_qwen2.5_7B_splithalf.csv'),
    ('TruthfulQA', 'qwen3-8B*', 'results/dim_angle_vs_n_qwen3_8B_splithalf.csv'),
    ('Refusal', 'llama3-8B-Instruct', 'results/dim_angle_refusal_llama3_8B_instruct_splithalf.csv'),
    ('Refusal', 'qwen2.5-7B-Instruct', 'results/dim_angle_refusal_qwen2.5_7B_splithalf.csv'),
    ('Refusal', 'gemma-2-9B-it', 'results/dim_angle_refusal_gemma2_9B_it_splithalf.csv'),
    ('Sycophancy', 'llama3-8B-Instruct', 'results/dim_angle_sycophancy_llama3_8B_instruct_splithalf.csv'),
    ('Sycophancy', 'qwen2.5-7B-Instruct', 'results/dim_angle_sycophancy_qwen2.5_7B_splithalf.csv'),
    ('Sycophancy', 'gemma-2-9B-it', 'results/dim_angle_sycophancy_gemma2_9B_it_splithalf.csv'),
]


fig, axes = plt.subplots(3, 3, figsize=(12, 10.2), sharey=False)
axes = axes.flatten()

for ax, (dataset, model, path) in zip(axes, panels):
    df = pd.read_csv(path)
    n_max = df['n'].max()

    C, r2 = fit_tan2_c(df['n'].values, df['mean'].values)
    is_flagged = r2 < 0.9
    color = COLOR_FLAG if is_flagged else COLOR_OK

    ax.fill_between(df['n'], df['min'], df['max'], color=color, alpha=0.18)
    ax.plot(df['n'], df['min'], color=color, linewidth=0.9, linestyle='--', alpha=0.8)
    ax.plot(df['n'], df['max'], color=color, linewidth=0.9, linestyle='--', alpha=0.8)
    ax.plot(df['n'], df['mean'], color=color, linewidth=1.8)

    title = f'{dataset} — {model}'
    if is_flagged:
        title += '  ⚠'
    ax.set_title(title, fontsize=9.5)
    fit_label = f'$C$={C:.1f}, $R^2$={r2:.2f}'
    ax.text(0.97, 0.95, fit_label, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color=color,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color, alpha=0.85, linewidth=0.8))

    ax.set_xlim(1, n_max)
    ax.set_ylim(0, max(80, df['max'].max() * 1.05))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5)

for ax in axes[6:]:
    ax.set_xlabel('sample size $n$ (per disjoint half)', fontsize=9)
for ax in [axes[0], axes[3], axes[6]]:
    ax.set_ylabel('split-half angle\n(degrees)', fontsize=9)

fig.suptitle('DiM direction split-half angle vs. subsample size, across models and datasets\n'
             r'(solid: mean; shaded/dashed: min–max over 200 draws; fit: $\tan^2(\theta_{split}/\sqrt{2}) \approx C/n$)',
             fontsize=11, y=1.015)
fig.text(0.5, -0.01,
         '*TruthfulQA column 3 shows qwen3-8B (no gemma-2-9B-it TruthfulQA run). '
         'Orange panels (⚠, $R^2$<0.9) show a poor 1/n fit — a signal of pool heterogeneity '
         '(e.g. subpopulations whose DiM directions disagree), not just sampling noise.',
         ha='center', fontsize=7.5, color='#555555', wrap=True)
fig.tight_layout()

out_path = 'figures/dim_angle_vs_n_multi_model.pdf'
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved to {out_path}")

print(f"\n{'panel':45s} {'C':>8s} {'R^2':>8s}")
for ax, (dataset, model, path) in zip(axes, panels):
    df = pd.read_csv(path)
    C, r2 = fit_tan2_c(df['n'].values, df['mean'].values)
    flag = ' <-- heterogeneous' if r2 < 0.9 else ''
    print(f"{dataset + ' / ' + model:45s} {C:8.2f} {r2:8.3f}{flag}")
