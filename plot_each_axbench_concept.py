# Save one plain-vs-reweighted projection-strength figure per axbench
# concept (20 total), using the already-computed
# reweighted_unpaired_axbench_concept{cid}_clean.npz files and the
# C/R^2/angle summary from run_all_axbench_concepts.py.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
OUT_DIR = os.path.join(HERE, 'figures', 'axbench_per_concept')
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_POS = '#2a78d6'
COLOR_NEG = '#eb6834'

summary = pd.read_csv(os.path.join(RESULTS_DIR, 'axbench_all_concepts_summary.csv'))

for _, row in summary.iterrows():
    cid = int(row['concept_id'])
    desc = row['desc']
    C, r2, angle = row['C'], row['R2'], row['angle_plain_vs_reweighted']

    npz_path = os.path.join(RESULTS_DIR, f'reweighted_unpaired_axbench_concept{cid}_clean.npz')
    data = np.load(npz_path)
    pos, neg = data['pos'], data['neg']
    plain_mu, mu = data['plain_mu'], data['mu']

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=False)

    for ax, (direction, label) in zip(axes, [(plain_mu, 'plain DiM'), (mu, 'reweighted')]):
        a_pos = pos @ direction
        a_neg = neg @ direction
        bins = np.linspace(min(a_pos.min(), a_neg.min()), max(a_pos.max(), a_neg.max()), 40)
        ax.hist(a_pos, bins=bins, color=COLOR_POS, alpha=0.75, label=f'concept-text (n={len(a_pos)})')
        ax.hist(a_neg, bins=bins, color=COLOR_NEG, alpha=0.75, label=f'shared background (n={len(a_neg)})')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r'$a_i = \hat\mu \cdot x_i$')
        if ax is axes[0]:
            ax.set_ylabel('count')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, linewidth=0.5, axis='y')

    fig.suptitle(f'axbench concept {cid}: {desc}\n'
                 f'$C$={C:.2f}, $R^2$={r2:.3f}, angle(plain, reweighted)={angle:.1f}°',
                 fontsize=10.5, y=1.05)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, f'concept{cid}.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

print(f"\nAll {len(summary)} concept figures saved to {OUT_DIR}")
