# Reweighted DiM for UNPAIRED pos/neg groups (e.g. refusal's harmful/harmless,
# which are independently sampled -- not CAA contrastive pairs with a 1:1
# correspondence). reweighted_dim.py assumes paired z_i = pos_i - neg_i;
# this version reweights each group's samples independently.
#
# For each group (pos, neg) and fold k:
#   a_i   = mu_hat^(-k) . x_i              (signed alignment to current dir)
#   tau_i^2 = || (I - mu mu^T) x_i ||^2     (out-of-direction noise)
#   shrink tau_i^2 toward the fold's mean
#   w_i = 1 / (tau_tilde_i^2 + lambda)      (no relu: both classes should
#                                             keep their full weight budget --
#                                             only the noise term discriminates;
#                                             sign-gating only made sense for
#                                             per-pair differences, where a
#                                             negative a_i means "this pair
#                                             disagrees with the population",
#                                             not for a single class's own
#                                             samples around their own mean)
# Then mu_hat = normalize(weighted_mean(pos) - weighted_mean(neg)).

import numpy as np
import argparse

from common import normalize, fold_assignment


def reweight_one_group(X, mu, fold_id, K, gamma, lam, c):
    """Return new weights (sum to 1) for one group's samples, given the
    current direction mu, using K-fold cross-fitting (fold k's weights use
    mu re-estimated on the other folds' weighted mean within this group)."""
    N = len(X)
    w = np.full(N, 1.0 / N)

    mu_minus_k = {}
    for k in range(K):
        mask = fold_id != k
        mu_minus_k[k] = mu  # use the shared current direction as the pivot

    a = np.empty(N)
    tau2 = np.empty(N)
    for k in range(K):
        mask = fold_id == k
        Xk = X[mask]
        proj_coef = Xk @ mu
        a[mask] = proj_coef
        resid = Xk - np.outer(proj_coef, mu)
        tau2[mask] = np.sum(resid ** 2, axis=1)

    tau2_tilde = (1 - gamma) * tau2 + gamma * tau2.mean()
    w_new = 1.0 / (tau2_tilde + lam)
    w_new = np.clip(w_new, 0, c / N)
    w_new = w_new / w_new.sum()
    return w_new


def reweighted_dim_unpaired(pos, neg, K=5, gamma=0.3, lam=0.01, c=5.0, T=10, seed=42, verbose=False):
    rng = np.random.RandomState(seed)
    fold_pos = fold_assignment(len(pos), K, rng)
    fold_neg = fold_assignment(len(neg), K, rng)

    mu = normalize(pos.mean(axis=0) - neg.mean(axis=0))
    plain_mu = mu.copy()

    w_pos = np.full(len(pos), 1.0 / len(pos))
    w_neg = np.full(len(neg), 1.0 / len(neg))

    for t in range(T):
        w_pos = reweight_one_group(pos, mu, fold_pos, K, gamma, lam, c)
        w_neg = reweight_one_group(neg, mu, fold_neg, K, gamma, lam, c)

        mu_pos = (w_pos[:, None] * pos).sum(axis=0)
        mu_neg = (w_neg[:, None] * neg).sum(axis=0)
        mu = normalize(mu_pos - mu_neg)

        if verbose:
            angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
            n_eff_pos = 1.0 / np.sum(w_pos ** 2)
            n_eff_neg = 1.0 / np.sum(w_neg ** 2)
            print(f"  iter {t+1}/{T}: angle-to-plain={angle:.2f} deg, "
                  f"N_eff_pos={n_eff_pos:.1f}/{len(pos)}, N_eff_neg={n_eff_neg:.1f}/{len(neg)}")

    return mu, w_pos, w_neg, plain_mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--pos_key', type=str, default='harmful')
    parser.add_argument('--neg_key', type=str, default='harmless')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--lam', type=float, default=0.01)
    parser.add_argument('--c', type=float, default=5.0)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_npz', type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    pos = data[args.pos_key]
    neg = data[args.neg_key]
    print(f"pos: {pos.shape}, neg: {neg.shape}")

    mu, w_pos, w_neg, plain_mu = reweighted_dim_unpaired(
        pos, neg, K=args.K, gamma=args.gamma, lam=args.lam, c=args.c, T=args.T,
        seed=args.seed, verbose=True)

    np.savez(args.out_npz, mu=mu, plain_mu=plain_mu, w_pos=w_pos, w_neg=w_neg, pos=pos, neg=neg)
    print(f"Saved to {args.out_npz}")


if __name__ == "__main__":
    main()
