# Project pos (harmful) and neg (harmless) activations separately onto the
# pooled DiM direction mu = normalize(mean(pos) - mean(neg)), and plot each
# group's own projection-strength distribution. This shows how well the
# direction actually separates the two classes (not a per-pair difference
# vector's projection, but each individual sample's own projection).

import os
import sys
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from common import setup_paper_style, COLOR_POS, COLOR_NEG
setup_paper_style()
import matplotlib.pyplot as plt

models = [
    ('llama3-8B-Instruct', 'features_refusal/llama3_8B_instruct_refusal_layer12_pos-5.npz'),
    ('qwen2.5-7B-Instruct', 'features_refusal/qwen2.5_7B_refusal_layer15_pos-1.npz'),
    ('gemma-2-9B-it', 'features_refusal/gemma2_9B_it_refusal_layer31_pos-1.npz'),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)

for ax, (tag, npz_path) in zip(axes, models):
    data = np.load(npz_path)
    pos_acts = data['harmful']
    neg_acts = data['harmless']

    mu = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    mu = mu / np.linalg.norm(mu)

    a_pos = pos_acts @ mu
    a_neg = neg_acts @ mu

    bins = np.linspace(min(a_pos.min(), a_neg.min()), max(a_pos.max(), a_neg.max()), 40)
    ax.hist(a_pos, bins=bins, color=COLOR_POS, alpha=0.75, label=f'harmful (pos)  (n={len(a_pos)})')
    ax.hist(a_neg, bins=bins, color=COLOR_NEG, alpha=0.75, label=f'harmless (neg)  (n={len(a_neg)})')

    # simple separation diagnostic: overlap of the two distributions
    overlap_lo = max(a_pos.min(), a_neg.min())
    overlap_hi = min(a_pos.max(), a_neg.max())
    frac_pos_below_neg_mean = (a_pos < a_neg.mean()).mean()
    frac_neg_above_pos_mean = (a_neg > a_pos.mean()).mean()

    ax.set_title(f'{tag}\n(misclassified-by-midpoint: '
                 f'{min(frac_pos_below_neg_mean, frac_neg_above_pos_mean)*100:.0f}%-ish overlap)', fontsize=9.5)
    ax.set_xlabel(r'$a_i = \hat\mu \cdot x_i$  (projection onto pooled DiM direction)')
    if ax is axes[0]:
        ax.set_ylabel('count')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5, axis='y')

fig.suptitle('Per-sample projection strength of harmful (pos) vs. harmless (neg) activations\n'
             'onto their own pooled DiM direction (Refusal dataset)',
             fontsize=11, y=1.06)
fig.tight_layout()

out_path = 'figures/pos_neg_strength_refusal.pdf'
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved to {out_path}")
