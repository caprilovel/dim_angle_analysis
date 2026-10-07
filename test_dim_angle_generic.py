# Generic version of test_dim_angle_sample.py for a single DIM direction
# (residual-stream-level pos/neg activations, e.g. refusal harmful/harmless,
# or sycophancy pos/neg), loaded from an .npz with 'pos'/'neg' (or
# 'harmful'/'harmless') arrays of shape (n_samples, hidden_size).
#
# Unlike test_dim_angle_sample.py (which operates per attention head on the
# TruthfulQA ITI pipeline), this works on a single already-extracted
# direction's source activations -- no probe-based head selection needed.
#
# Two modes:
#   --mode full (default): angle between an n-subsample's DiM direction and
#     the FULL-POOL DiM direction. Biased as n approaches the pool size --
#     the subsample and the full-data direction share most of their samples,
#     so the angle is driven artificially toward 0 (most visible when
#     n_max == pool size, e.g. sycophancy's n=500 with 500-sample pools).
#   --mode split_half: angle between TWO DISJOINT n-subsamples drawn from
#     the same pool (so max usable n is pool_size // 2). This never
#     compares a subsample against itself, so it has no such degeneracy.
#     Under a small-angle approximation, the single-estimate-vs-true-mean
#     angle theta satisfies E[split_half_angle] ~= sqrt(2) * E[theta], so a
#     split-half curve also lets you back out the single-estimate error
#     without ever knowing the true direction. It additionally lets you fit
#     tan^2(theta_split/sqrt(2)) ~= C/n per model/dataset -- C is an
#     effective inverse-SNR; see fit_dim_angle_c.py.

import numpy as np
import argparse
import os
import csv

from common import angle_between


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--pos_key', type=str, default='pos')
    parser.add_argument('--neg_key', type=str, default='neg')
    parser.add_argument('--mode', type=str, default='split_half', choices=['full', 'split_half'])
    parser.add_argument('--n_min', type=int, default=1)
    parser.add_argument('--n_max', type=int, default=500)
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--log_points', type=int, default=0,
                         help='if >0, use this many log-spaced n values instead of every integer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    data = np.load(args.npz_path)
    pos_acts = data[args.pos_key]
    neg_acts = data[args.neg_key]
    print(f"[{args.tag}] pos_acts: {pos_acts.shape}, neg_acts: {neg_acts.shape}, mode={args.mode}")

    n_pos_max = len(pos_acts)
    n_neg_max = len(neg_acts)

    if args.mode == 'full':
        full_direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
        n_max_usable = min(args.n_max, n_pos_max, n_neg_max)
    else:
        # split_half needs two disjoint n-subsets from each pool
        n_max_usable = min(args.n_max, n_pos_max // 2, n_neg_max // 2)

    if n_max_usable < args.n_max:
        print(f"Note: capping n_max to {n_max_usable} (available pos={n_pos_max}, neg={n_neg_max}, mode={args.mode})")

    if args.log_points > 0:
        n_values = np.unique(np.round(
            np.geomspace(args.n_min, n_max_usable, args.log_points)
        ).astype(int))
        n_values = n_values[(n_values >= args.n_min) & (n_values <= n_max_usable)]
    else:
        n_values = np.arange(args.n_min, n_max_usable + 1)

    rows = []
    for n in n_values:
        n = int(n)
        angles = np.empty(args.n_trials)
        for t in range(args.n_trials):
            if args.mode == 'full':
                pos_idx = rng.choice(n_pos_max, size=n, replace=False)
                neg_idx = rng.choice(n_neg_max, size=n, replace=False)
                sub_direction = pos_acts[pos_idx].mean(axis=0) - neg_acts[neg_idx].mean(axis=0)
                angles[t] = angle_between(full_direction, sub_direction)
            else:
                pos_idx = rng.choice(n_pos_max, size=2 * n, replace=False)
                neg_idx = rng.choice(n_neg_max, size=2 * n, replace=False)
                dir_a = pos_acts[pos_idx[:n]].mean(axis=0) - neg_acts[neg_idx[:n]].mean(axis=0)
                dir_b = pos_acts[pos_idx[n:]].mean(axis=0) - neg_acts[neg_idx[n:]].mean(axis=0)
                angles[t] = angle_between(dir_a, dir_b)
        rows.append({
            'n': n,
            'min': angles.min(),
            'max': angles.max(),
            'mean': angles.mean(),
            'median': np.median(angles),
            'std': angles.std(),
            'p5': np.percentile(angles, 5),
            'p95': np.percentile(angles, 95),
        })
        if n % 50 == 0 or n == args.n_min or n == n_max_usable:
            print(f"n={n:5d}  min={rows[-1]['min']:.2f}  mean={rows[-1]['mean']:.2f}  max={rows[-1]['max']:.2f}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-n angle stats to {args.out_csv}")
    print(f"(mode={args.mode}, pos pool={n_pos_max}, neg pool={n_neg_max}, dim={pos_acts.shape[1]}, n_trials={args.n_trials})")


if __name__ == "__main__":
    main()
