# Reweighted difference-in-means (DiM) estimator.
#
# Standard DiM is mu_hat = normalize(mean_i z_i), which implicitly weights
# every pair z_i = pos_i - neg_i equally. If the pool is heterogeneous --
# some pairs' z_i point in a very different direction than the population
# mean, e.g. because they come from a different sub-behavior or are noisy/
# mislabeled -- equal weighting lets those pairs drag the estimate around,
# which is exactly the failure mode the split-half tan^2(theta) ~ C/n fit
# flags (R^2 << 1, slope far from -1: see fit_dim_angle_c.py).
#
# This implements a soft, self-consistent downweighting scheme:
#   1. K-fold cross-fitting: estimate each fold's weights using a direction
#      fit on the OTHER folds only (mu_hat^(-k)), so a pair never
#      influences its own weight -- avoids overfitting the weights to
#      individual pairs' noise.
#   2. For pair i in fold k, project z_i onto mu_hat^(-k):
#        a_i   = mu_hat^(-k) . z_i          (signed alignment)
#        tau_i^2 = || (I - mu mu^T) z_i ||^2  (out-of-direction "noise" energy)
#      Shrink tau_i^2 toward the fold's mean (single-sample variance
#      estimates are themselves noisy) and form
#        w_i = max(a_i, 0) / (tau_tilde_i^2 + lambda)
#      i.e. upweight pairs that (a) agree in sign with the current
#      direction and (b) have small out-of-direction noise; zero out pairs
#      that point the "wrong way" entirely.
#   3. Clip weights to c/N, renormalize, recompute mu_hat, repeat T times.
#
# Output: mu_hat (unit vector) and the final per-pair weights.

import numpy as np
import argparse

from common import normalize, fold_assignment


def reweighted_dim(Z, K=5, gamma=0.3, lam=0.01, c=5.0, T=10, seed=42, init_mu=None, verbose=False):
    """
    Z: (N, d) array of pair-difference vectors z_i = pos_i - neg_i.
    Returns (mu_hat, w) where mu_hat is a unit vector and w are the final
    per-pair weights (sum to 1).
    """
    N, d = Z.shape
    rng = np.random.RandomState(seed)
    fold_id = fold_assignment(N, K, rng)

    if init_mu is not None:
        mu = normalize(init_mu.copy())
    else:
        mu = normalize(Z.mean(axis=0))

    w = np.full(N, 1.0 / N)

    for t in range(T):
        # mu_hat^(-k): direction fit on all folds except k, using the
        # CURRENT weights restricted to those folds
        mu_minus_k = {}
        for k in range(K):
            mask = fold_id != k
            w_k = w[mask]
            w_k = w_k / w_k.sum()
            mu_minus_k[k] = normalize((w_k[:, None] * Z[mask]).sum(axis=0))

        a = np.empty(N)
        tau2 = np.empty(N)
        for k in range(K):
            mask = fold_id == k
            mu_k = mu_minus_k[k]
            Zk = Z[mask]
            a[mask] = Zk @ mu_k
            proj = np.outer(Zk @ mu_k, mu_k)
            resid = Zk - proj
            tau2[mask] = np.sum(resid ** 2, axis=1)

        tau2_tilde = (1 - gamma) * tau2 + gamma * tau2.mean()

        w_new = np.clip(a, 0, None) / (tau2_tilde + lam)
        w_new = np.clip(w_new, 0, c / N)
        if w_new.sum() == 0:
            raise RuntimeError("all weights collapsed to zero -- check lambda/gamma/c")
        w_new = w_new / w_new.sum()
        w = w_new

        mu = normalize((w[:, None] * Z).sum(axis=0))

        if verbose:
            angle_to_prev = np.degrees(np.arccos(np.clip(mu @ normalize(Z.mean(axis=0)), -1, 1)))
            print(f"  iter {t+1}/{T}: effective N_eff={1.0/np.sum(w**2):.1f}, "
                  f"angle-to-plain-DiM={angle_to_prev:.2f} deg, "
                  f"max_w={w.max():.4f} (cap={c/N:.4f})")

    return mu, w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--pos_key', type=str, default='pos')
    parser.add_argument('--neg_key', type=str, default='neg')
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
    n = min(len(pos), len(neg))
    Z = pos[:n] - neg[:n]
    print(f"Loaded {n} pairs, dim={Z.shape[1]}")

    plain_mu = normalize(Z.mean(axis=0))

    mu, w = reweighted_dim(Z, K=args.K, gamma=args.gamma, lam=args.lam, c=args.c, T=args.T,
                            seed=args.seed, verbose=True)

    angle = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
    print(f"\nFinal angle between reweighted mu and plain DiM: {angle:.2f} deg")
    print(f"Weight stats: min={w.min():.2e}, max={w.max():.4f}, "
          f"effective N_eff=1/sum(w^2)={1.0/np.sum(w**2):.1f} (of N={n})")

    np.savez(args.out_npz, mu=mu, w=w, plain_mu=plain_mu, Z=Z)
    print(f"Saved to {args.out_npz}")


if __name__ == "__main__":
    main()
