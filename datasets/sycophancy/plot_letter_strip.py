# Strip/histogram plot of per-pair direction strength a_i = mu_pooled . z_i,
# colored by the pos_answer letter ("(A)" vs "(B)"), across all three models'
# sycophancy activations. Shows directly how much of each pool's variance in
# projected strength is explained by the letter nuisance variable.

import os
import sys
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from common import setup_paper_style, ANTHROPIC_AI_RISK_DIR
setup_paper_style()
import matplotlib.pyplot as plt

sys.path.insert(0, ANTHROPIC_AI_RISK_DIR)
from data import make_contrastive_pairs

COLOR_A = '#2a78d6'
COLOR_B = '#eb6834'

pairs = make_contrastive_pairs('sycophancy_on_political_typology_quiz', n_samples=500, seed=42)
pos_letters = np.array([p.pos_answer.strip().strip('()') for p in pairs])

models = [
    ('llama3-8B-Instruct', 'features_sycophancy/llama3_8B_instruct_sycophancy_layer12.npz'),
    ('qwen2.5-7B-Instruct', 'features_sycophancy/qwen2.5_7B_sycophancy_layer15.npz'),
    ('gemma-2-9B-it', 'features_sycophancy/gemma2_9B_it_sycophancy_layer31.npz'),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)

for ax, (tag, npz_path) in zip(axes, models):
    data = np.load(npz_path)
    Z = data['pos'] - data['neg']
    mu_pooled = Z.mean(axis=0)
    mu_pooled = mu_pooled / np.linalg.norm(mu_pooled)
    a = Z @ mu_pooled

    maskA = pos_letters == 'A'
    maskB = pos_letters == 'B'

    bins = np.linspace(a.min(), a.max(), 40)
    ax.hist(a[maskA], bins=bins, color=COLOR_A, alpha=0.75, label=f'pos_answer = (A)  (n={maskA.sum()})')
    ax.hist(a[maskB], bins=bins, color=COLOR_B, alpha=0.75, label=f'pos_answer = (B)  (n={maskB.sum()})')
    ax.axvline(0, color='#555555', linewidth=1, linestyle='--')

    frac_neg = (a < 0).mean()
    ax.set_title(f'{tag}\n({frac_neg*100:.0f}% of all pairs project negative)', fontsize=10)
    ax.set_xlabel(r'$a_i = \hat\mu_{pooled} \cdot z_i$  (projection strength)')
    if ax is axes[0]:
        ax.set_ylabel('count')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5, axis='y')

fig.suptitle('Per-pair projection strength onto the pooled DiM direction, split by answer-letter token\n'
             '(Sycophancy / sycophancy_on_political_typology_quiz, same 500 pairs across models)',
             fontsize=11, y=1.06)
fig.tight_layout()

out_path = 'figures/letter_strip_sycophancy.pdf'
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved to {out_path}")
