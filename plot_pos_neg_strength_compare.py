# Compare: project harmful (pos) / harmless (neg) activations onto (a) the
# plain DiM direction vs (b) the reweighted direction, side by side, to see
# whether reweighting changes the separation at all. On a clean dataset
# like Refusal, we expect almost no visible difference (reweighted_dim_
# unpaired.py already showed the two directions are <1.3 deg apart with
# N_eff ~= N for all three models).

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

COLOR_POS = '#2a78d6'
COLOR_NEG = '#eb6834'

models = [
    ('llama3-8B-Instruct', 'results/reweighted_unpaired_llama3_refusal.npz'),
    ('qwen2.5-7B-Instruct', 'results/reweighted_unpaired_qwen25_refusal.npz'),
    ('gemma-2-9B-it', 'results/reweighted_unpaired_gemma2_refusal.npz'),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)

for col, (tag, npz_path) in enumerate(models):
    data = np.load(npz_path)
    pos, neg = data['pos'], data['neg']
    plain_mu, mu = data['plain_mu'], data['mu']

    angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))

    for row, (direction, label) in enumerate([(plain_mu, 'plain DiM'), (mu, 'reweighted')]):
        ax = axes[row, col]
        a_pos = pos @ direction
        a_neg = neg @ direction
        bins = np.linspace(min(a_pos.min(), a_neg.min()), max(a_pos.max(), a_neg.max()), 40)
        ax.hist(a_pos, bins=bins, color=COLOR_POS, alpha=0.75, label=f'harmful (n={len(a_pos)})')
        ax.hist(a_neg, bins=bins, color=COLOR_NEG, alpha=0.75, label=f'harmless (n={len(a_neg)})')

        if row == 0:
            ax.set_title(f'{tag}\n{label} direction', fontsize=10)
        else:
            ax.set_title(f'{label} direction  (angle to plain: {angle:.2f}°)', fontsize=9.5)
        ax.set_xlabel(r'$a_i = \hat\mu \cdot x_i$')
        if col == 0:
            ax.set_ylabel('count')
        ax.legend(frameon=False, fontsize=7.5, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, linewidth=0.5, axis='y')

fig.suptitle('Refusal: harmful vs. harmless projection strength, plain DiM (top) vs. reweighted DiM (bottom)',
             fontsize=12, y=1.01)
fig.tight_layout()

out_path = 'figures/pos_neg_strength_refusal_compare.pdf'
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved to {out_path}")
