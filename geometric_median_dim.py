# Robust mean estimation via grouping + geometric median, applied to DiM.
#
# Based on the "median-of-means via geometric median" construction from
# Minsker, "Geometric median and robust estimation in Banach spaces"
# (and surveyed in "The Geometric Median and Applications to Robust Mean
# Estimation"): split N i.i.d. samples into k disjoint groups, take each
# group's ordinary mean (a "weak" estimator that is already unbiased but
# can have heavy-tailed/outlier-corrupted error), then take the geometric
# median (the point minimizing the sum of Euclidean distances, not the
# per-coordinate median) of those k group means. This estimator tolerates
# up to ~k/2 of the groups being arbitrarily corrupted while the error of
# the remaining groups' means still averages out -- unlike the plain
# sample mean, which a single extreme outlier can drag arbitrarily far.
#
# Two DiM variants, mirroring reweighted_dim.py / reweighted_dim_unpaired.py:
#   - gm_dim_paired(Z, ...):   Z = pos_i - neg_i (paired differences);
#                              one robust mean over Z, mu = normalize(gm(Z)).
#   - gm_dim_unpaired(pos, neg, ...): independent groups; robust mean of
#                              pos and of neg separately, then
#                              mu = normalize(gm(pos) - gm(neg)).
#
# The geometric median has no closed form; we solve it with Weiszfeld's
# algorithm (iteratively reweighted average, weights = 1/distance to the
# current estimate), which converges to the global minimum since the sum
# of Euclidean norms is convex.
#
# Usage:
#   python geometric_median_dim.py --npz_path features_refusal/....npz \
#       --mode unpaired --pos_key harmful --neg_key harmless \
#       --k_groups 9 --out_npz results/gm_dim_refusal.npz

import numpy as np
import argparse

from common import normalize


def geometric_median(points, tol=1e-8, max_iter=500, eps=1e-12):
    """Weiszfeld's algorithm: the point minimizing sum_i ||points[i] - x||.

    points: (k, d) array. Returns a (d,) array.
    """
    points = np.asarray(points, dtype=float)
    x = points.mean(axis=0)

    for _ in range(max_iter):
        dist = np.linalg.norm(points - x, axis=1)

        # If x lands exactly on one of the points, Weiszfeld's update is
        # undefined there; nudge by treating that point's weight as
        # dominant (standard fix, e.g. Vardi-Zhang) -- in practice with
        # continuous activation vectors this is never hit, but guard anyway.
        zero_mask = dist < eps
        if np.any(zero_mask):
            if zero_mask.sum() == len(points):
                return x
            dist = np.where(zero_mask, eps, dist)

        w = 1.0 / dist
        x_new = (w[:, None] * points).sum(axis=0) / w.sum()

        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new

    return x


def _group_means(X, k_groups, rng):
    """Shuffle X and split into k_groups (near-)equal groups; return their
    per-group means, shape (k_groups, d)."""
    N = len(X)
    idx = np.arange(N)
    rng.shuffle(idx)
    groups = np.array_split(idx, k_groups)
    return np.stack([X[g].mean(axis=0) for g in groups], axis=0)


def gm_dim_paired(Z, k_groups=9, seed=42, verbose=False):
    """Robust DiM for paired differences z_i = pos_i - neg_i.

    Returns (mu_hat, plain_mu, group_means).
    """
    rng = np.random.RandomState(seed)
    plain_mu = normalize(Z.mean(axis=0))

    group_means = _group_means(Z, k_groups, rng)
    gm = geometric_median(group_means)
    mu = normalize(gm)

    if verbose:
        angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
        print(f"  k_groups={k_groups}, N={len(Z)} -> {len(group_means)} group means, "
              f"angle-to-plain-DiM={angle:.2f} deg")

    return mu, plain_mu, group_means


def gm_dim_unpaired(pos, neg, k_groups=9, seed=42, verbose=False):
    """Robust DiM for independent pos/neg groups.

    Returns (mu_hat, plain_mu, pos_group_means, neg_group_means).
    """
    rng = np.random.RandomState(seed)
    plain_mu = normalize(pos.mean(axis=0) - neg.mean(axis=0))

    pos_group_means = _group_means(pos, k_groups, rng)
    neg_group_means = _group_means(neg, k_groups, rng)
    gm_pos = geometric_median(pos_group_means)
    gm_neg = geometric_median(neg_group_means)
    mu = normalize(gm_pos - gm_neg)

    if verbose:
        angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
        print(f"  k_groups={k_groups}, N_pos={len(pos)}, N_neg={len(neg)} -> "
              f"{len(pos_group_means)}/{len(neg_group_means)} group means, "
              f"angle-to-plain-DiM={angle:.2f} deg")

    return mu, plain_mu, pos_group_means, neg_group_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--mode', choices=['paired', 'unpaired'], required=True)
    parser.add_argument('--pos_key', type=str, default='pos')
    parser.add_argument('--neg_key', type=str, default='neg')
    parser.add_argument('--k_groups', type=int, default=9,
                         help='number of groups to split each pool into before '
                              'taking the geometric median of group means')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_npz', type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    pos = data[args.pos_key]
    neg = data[args.neg_key]

    if args.mode == 'paired':
        n = min(len(pos), len(neg))
        Z = pos[:n] - neg[:n]
        print(f"Loaded {n} pairs, dim={Z.shape[1]}")

        mu, plain_mu, group_means = gm_dim_paired(
            Z, k_groups=args.k_groups, seed=args.seed, verbose=True)

        angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
        print(f"\nFinal angle between geometric-median DiM and plain DiM: {angle:.2f} deg")
        np.savez(args.out_npz, mu=mu, plain_mu=plain_mu, group_means=group_means, Z=Z)
    else:
        print(f"pos: {pos.shape}, neg: {neg.shape}")

        mu, plain_mu, pos_group_means, neg_group_means = gm_dim_unpaired(
            pos, neg, k_groups=args.k_groups, seed=args.seed, verbose=True)

        angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
        print(f"\nFinal angle between geometric-median DiM and plain DiM: {angle:.2f} deg")
        np.savez(args.out_npz, mu=mu, plain_mu=plain_mu,
                 pos_group_means=pos_group_means, neg_group_means=neg_group_means,
                 pos=pos, neg=neg)

    print(f"Saved to {args.out_npz}")


if __name__ == "__main__":
    main()
