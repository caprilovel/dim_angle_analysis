# Given a FIXED reference direction (here: the reweighted mu_hat), look at
# how individual pairs' z_i = pos_i - neg_i project onto it:
#   a_i   = mu_hat . z_i         (signed strength/alignment)
#   tau_i = || (I - mu mu^T) z_i ||   (out-of-direction noise magnitude)
#
# This visualizes the heterogeneity directly: a healthy i.i.d. pool should
# show a_i unimodal and mostly positive (most pairs agree with the
# population direction); a bimodal or heavy-negative-tail distribution
# means a subgroup of pairs point the "wrong way" relative to the rest --
# exactly what pulls the plain-mean DiM off course and what the reweighting
# scheme downweights.

import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


def normalize(v):
    return v / np.linalg.norm(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reweighted_npz', type=str, required=True,
                         help='output of reweighted_dim.py (has mu, plain_mu, Z, w)')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--out_pdf', type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.reweighted_npz)
    mu = data['mu']
    plain_mu = data['plain_mu']
    Z = data['Z']
    w = data['w']
    N = Z.shape[0]

    a_reweighted = Z @ mu
    proj = np.outer(a_reweighted, mu)
    resid = Z - proj
    tau_reweighted = np.linalg.norm(resid, axis=1)

    a_plain = Z @ plain_mu
    proj_p = np.outer(a_plain, plain_mu)
    resid_p = Z - proj_p
    tau_plain = np.linalg.norm(resid_p, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.hist(a_plain, bins=40, color='#2a78d6', alpha=0.75)
    ax.axvline(0, color='#555555', linewidth=1, linestyle='--')
    frac_neg = (a_plain < 0).mean()
    ax.set_title(f'Projection onto PLAIN DiM direction\n({frac_neg*100:.1f}% of pairs have a_i < 0)', fontsize=10)
    ax.set_xlabel('$a_i = \\mu_{plain} \\cdot z_i$')
    ax.set_ylabel('count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    ax.hist(a_reweighted, bins=40, color='#1baf7a', alpha=0.75)
    ax.axvline(0, color='#555555', linewidth=1, linestyle='--')
    frac_neg_rw = (a_reweighted < 0).mean()
    ax.set_title(f'Projection onto REWEIGHTED direction\n({frac_neg_rw*100:.1f}% of pairs have a_i < 0)', fontsize=10)
    ax.set_xlabel('$a_i = \\hat\\mu_{rw} \\cdot z_i$')
    ax.set_ylabel('count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 0]
    sc = ax.scatter(a_plain, tau_plain, c=w, cmap='viridis', s=14, alpha=0.75)
    ax.axvline(0, color='#555555', linewidth=1, linestyle='--')
    ax.set_xlabel('$a_i$ (alignment, plain-DiM axis)')
    ax.set_ylabel(r'$\tau_i$ (out-of-direction noise)')
    ax.set_title('Per-pair alignment vs. noise\n(color = final reweight-algorithm weight)', fontsize=10)
    plt.colorbar(sc, ax=ax, label='weight $w_i$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    order = np.argsort(-w)
    ax.plot(np.arange(1, N + 1), np.sort(w)[::-1], color='#eb6834', linewidth=1.8)
    ax.axhline(1.0 / N, color='gray', linewidth=1, linestyle=':', label=f'uniform (1/N={1.0/N:.4f})')
    n_eff = 1.0 / np.sum(w ** 2)
    ax.set_xlabel('pair rank (sorted by weight, descending)')
    ax.set_ylabel('weight $w_i$')
    ax.set_title(f'Reweighting profile\n(effective N_eff = {n_eff:.1f} of {N})', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    angle_plain_rw = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
    fig.suptitle(f'{args.tag}: pair-level direction strength (angle plain vs. reweighted = {angle_plain_rw:.1f}°)',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(args.out_pdf, bbox_inches='tight')
    print(f"Saved to {args.out_pdf}")

    print(f"\nplain DiM:       {(a_plain < 0).mean()*100:.1f}% negative-projection pairs")
    print(f"reweighted dir:  {(a_reweighted < 0).mean()*100:.1f}% negative-projection pairs")
    print(f"angle(plain, reweighted) = {angle_plain_rw:.2f} deg")
    print(f"N_eff = {n_eff:.1f} / {N}  ({n_eff/N*100:.1f}% of pairs carry the effective weight)")


if __name__ == "__main__":
    main()
