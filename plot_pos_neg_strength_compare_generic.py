# Generic version of plot_pos_neg_strength_compare.py: pass a list of
# (tag, npz_path) pairs (output of reweighted_dim_unpaired.py) and a title/
# output path, instead of hardcoding the refusal models.

import argparse
import numpy as np
from common import setup_paper_style, COLOR_POS, COLOR_NEG
setup_paper_style()
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, nargs='+', required=True)
    parser.add_argument('--npz', type=str, nargs='+', required=True)
    parser.add_argument('--pos_label', type=str, default='pos')
    parser.add_argument('--neg_label', type=str, default='neg')
    parser.add_argument('--suptitle', type=str, required=True)
    parser.add_argument('--out_pdf', type=str, required=True)
    args = parser.parse_args()
    assert len(args.tag) == len(args.npz)

    n_models = len(args.tag)
    fig, axes = plt.subplots(2, n_models, figsize=(4.6 * n_models, 8), sharey=False)
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, (tag, npz_path) in enumerate(zip(args.tag, args.npz)):
        data = np.load(npz_path)
        pos, neg = data['pos'], data['neg']
        plain_mu, mu = data['plain_mu'], data['mu']

        angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))

        for row, (direction, label) in enumerate([(plain_mu, 'plain DiM'), (mu, 'reweighted')]):
            ax = axes[row, col]
            a_pos = pos @ direction
            a_neg = neg @ direction
            bins = np.linspace(min(a_pos.min(), a_neg.min()), max(a_pos.max(), a_neg.max()), 40)
            ax.hist(a_pos, bins=bins, color=COLOR_POS, alpha=0.75, label=f'{args.pos_label} (n={len(a_pos)})')
            ax.hist(a_neg, bins=bins, color=COLOR_NEG, alpha=0.75, label=f'{args.neg_label} (n={len(a_neg)})')

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

    fig.suptitle(args.suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(args.out_pdf, bbox_inches='tight')
    print(f"Saved to {args.out_pdf}")

    print(f"\n{'model':20s} {'angle plain-vs-rw':>18s}")
    for tag, npz_path in zip(args.tag, args.npz):
        data = np.load(npz_path)
        angle = np.degrees(np.arccos(np.clip(data['mu'] @ data['plain_mu'], -1, 1)))
        print(f"{tag:20s} {angle:18.2f}")


if __name__ == "__main__":
    main()
