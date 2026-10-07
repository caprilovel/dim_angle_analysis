# Sycophancy version of the plain-vs-reweighted comparison, using the
# PAIRED reweighting (reweighted_dim.py), which operates on z_i = pos_i -
# neg_i and is the version that actually detects the letter-driven
# heterogeneity (the unpaired version, reweighted_dim_unpaired.py, treats
# pos/neg as independent groups and can't see the pair-level structure that
# the letter confound creates -- it produced <3 deg shifts on this same
# data, vs. 30-78 deg here).
#
# Projects the ORIGINAL pos/neg activations (not just Z) onto both the
# plain DiM direction and the paired-reweighted direction, so the panels
# are directly comparable to the refusal/TruthfulQA figures.

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

COLOR_POS = '#2a78d6'
COLOR_NEG = '#eb6834'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, nargs='+', required=True)
    parser.add_argument('--source_npz', type=str, nargs='+', required=True,
                         help='original dump_sycophancy_activations.py output (has pos, neg)')
    parser.add_argument('--reweighted_npz', type=str, nargs='+', required=True,
                         help='reweighted_dim.py output (has mu, plain_mu, Z, w)')
    parser.add_argument('--out_pdf', type=str, required=True)
    args = parser.parse_args()
    n_models = len(args.tag)

    fig, axes = plt.subplots(2, n_models, figsize=(4.6 * n_models, 8), sharey=False)
    if n_models == 1:
        axes = axes.reshape(2, 1)

    print(f"{'model':20s} {'angle plain-vs-rw':>18s} {'N_eff':>10s}")

    for col, (tag, src_path, rw_path) in enumerate(zip(args.tag, args.source_npz, args.reweighted_npz)):
        src = np.load(src_path)
        pos, neg = src['pos'], src['neg']

        rw = np.load(rw_path)
        plain_mu, mu, w = rw['plain_mu'], rw['mu'], rw['w']

        angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
        n_eff = 1.0 / np.sum(w ** 2)
        print(f"{tag:20s} {angle:18.2f} {n_eff:10.1f} / {len(w)}")

        for row, (direction, label) in enumerate([(plain_mu, 'plain DiM'), (mu, 'reweighted')]):
            ax = axes[row, col]
            a_pos = pos @ direction
            a_neg = neg @ direction
            bins = np.linspace(min(a_pos.min(), a_neg.min()), max(a_pos.max(), a_neg.max()), 40)
            ax.hist(a_pos, bins=bins, color=COLOR_POS, alpha=0.75, label=f'sycophantic (n={len(a_pos)})')
            ax.hist(a_neg, bins=bins, color=COLOR_NEG, alpha=0.75, label=f'non-syco. (n={len(a_neg)})')

            if row == 0:
                ax.set_title(f'{tag}\n{label} direction', fontsize=10)
            else:
                ax.set_title(f'{label} direction  (angle to plain: {angle:.1f}°, N_eff={n_eff:.0f})', fontsize=9)
            ax.set_xlabel(r'$a_i = \hat\mu \cdot x_i$')
            if col == 0:
                ax.set_ylabel('count')
            ax.legend(frameon=False, fontsize=7.5, loc='upper left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, linewidth=0.5, axis='y')

    fig.suptitle('Sycophancy: per-sample projection strength, plain DiM (top) vs. PAIRED-reweighted DiM (bottom)\n'
                 '(reweighting operates on z_i = pos_i - neg_i, which can see the letter-driven split; '
                 'unpaired per-group reweighting cannot)',
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out_pdf, bbox_inches='tight')
    print(f"\nSaved to {args.out_pdf}")


if __name__ == "__main__":
    main()
