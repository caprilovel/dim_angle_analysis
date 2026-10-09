# Verify that the pos_answer letter ("(A)" vs "(B)") -- not sycophancy
# semantics -- is the dominant confound behind qwen2.5-7B-Instruct's
# anomalously bad split_half fit on the sycophancy dataset (R^2 = -0.06).
#
# CAA extracts the activation AT the answer-letter token, so the activation
# naturally encodes "is this token A or B" with large magnitude, independent
# of whether that letter happens to be the sycophantic answer. Since which
# letter is sycophantic is assigned essentially at random per question, this
# acts as a near-orthogonal nuisance direction that splits the pool into two
# almost-disjoint clusters -- exactly the bimodal a_i histogram we saw.
#
# This script re-uses the already-extracted activations (no model rerun):
# it stratifies the 500 pairs by pos_answer letter and reruns the
# split-half C/R^2 diagnostic WITHIN each stratum. If letter is the
# confound, each stratum alone should fit the classic 1/n curve much better
# than the pooled (unstratified) data.

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from common import angle_between, fit_tan2_c, ANTHROPIC_AI_RISK_DIR

sys.path.insert(0, ANTHROPIC_AI_RISK_DIR)
from data import make_contrastive_pairs


def split_half_sweep(Z, n_max, n_trials=200, seed=42):
    rng = np.random.default_rng(seed)
    N = len(Z)
    rows = []
    for n in range(1, n_max + 1):
        angles = np.empty(n_trials)
        for t in range(n_trials):
            idx = rng.choice(N, size=2 * n, replace=False)
            dir_a = Z[idx[:n]].mean(axis=0)
            dir_b = Z[idx[n:2 * n]].mean(axis=0)
            angles[t] = angle_between(dir_a, dir_b)
        rows.append({'n': n, 'mean': angles.mean(), 'min': angles.min(), 'max': angles.max()})
    return pd.DataFrame(rows)


def main():
    pairs = make_contrastive_pairs('sycophancy_on_political_typology_quiz', n_samples=500, seed=42)
    pos_letters = np.array([p.pos_answer.strip().strip('()') for p in pairs])

    data = np.load('features_sycophancy/qwen2.5_7B_sycophancy_layer15.npz')
    pos, neg = data['pos'], data['neg']
    Z_all = pos - neg

    print("=== Pooled (unstratified) -- reference, matches earlier bad fit ===")
    df_pooled = split_half_sweep(Z_all, n_max=100, n_trials=200)
    C, r2 = fit_tan2_c(df_pooled['n'].values, df_pooled['mean'].values)
    print(f"  N={len(Z_all)}  C={C:.2f}  R^2={r2:.3f}")

    for letter in ['A', 'B']:
        mask = pos_letters == letter
        Z_sub = Z_all[mask]
        n_max = len(Z_sub) // 2
        print(f"\n=== Stratum: pos_answer = ({letter})  (N={len(Z_sub)}) ===")
        df = split_half_sweep(Z_sub, n_max=n_max, n_trials=200)
        C, r2 = fit_tan2_c(df['n'].values, df['mean'].values)
        print(f"  C={C:.2f}  R^2={r2:.3f}")
        df.to_csv(f'results/dim_angle_sycophancy_qwen2.5_7B_letter{letter}_splithalf.csv', index=False)

    print("\nIf letter is the confound: both strata should show R^2 >> pooled R^2.")


if __name__ == "__main__":
    main()
