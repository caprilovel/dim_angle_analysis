# Batch-run the full axbench concept pipeline (extract -> clean outliers ->
# split-half C/R^2 fit -> unpaired reweighting) for all 20 concepts in the
# expanded_9b_l31 cache, and summarize results in one table.

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO_ROOT)

from common import angle_between, fit_tan2_c, AXBENCH_CACHE_PATH
from dump_axbench_concept import load_axbench_cache
from reweighted_dim_unpaired import reweighted_dim_unpaired

CACHE_PATH = AXBENCH_CACHE_PATH
FEATURES_DIR = os.path.join(REPO_ROOT, 'features_axbench')
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NORM_OUTLIER_THRESHOLD = 1000.0  # see dump_axbench_concept.py diagnosis: ~6 background
                                   # samples have norm ~7636 vs a normal median of ~317-500


def split_half_sweep(Z_pos, Z_neg, n_max, n_trials=200, seed=42):
    """split-half on UNPAIRED pos/neg groups: draw 2n from pos and 2n from
    neg, split each in half, form two independent DiM directions, measure
    the angle between them."""
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(Z_pos), len(Z_neg)
    rows = []
    for n in range(1, n_max + 1):
        angles = np.empty(n_trials)
        for t in range(n_trials):
            pos_idx = rng.choice(n_pos, size=2 * n, replace=False)
            neg_idx = rng.choice(n_neg, size=2 * n, replace=False)
            dir_a = Z_pos[pos_idx[:n]].mean(axis=0) - Z_neg[neg_idx[:n]].mean(axis=0)
            dir_b = Z_pos[pos_idx[n:]].mean(axis=0) - Z_neg[neg_idx[n:]].mean(axis=0)
            angles[t] = angle_between(dir_a, dir_b)
        rows.append({'n': n, 'mean': angles.mean(), 'min': angles.min(), 'max': angles.max()})
    return pd.DataFrame(rows)


def main():
    hidden, labels, concept_ids, prompts = load_axbench_cache(CACHE_PATH)

    neg_mask_all = labels == 0
    pos_mask_all = labels == 1
    concepts = sorted(set(concept_ids[pos_mask_all].tolist()))
    print(f"Found {len(concepts)} concepts: {concepts}")

    # one representative descriptive label per concept, from its first prompt
    concept_desc = {}
    for cid in concepts:
        idx = np.where(pos_mask_all & (concept_ids == cid))[0][0]
        concept_desc[cid] = prompts[idx][:60]

    summary_rows = []

    for cid in concepts:
        pos_mask = pos_mask_all & (concept_ids == cid)
        pos = hidden[pos_mask]
        neg = hidden[neg_mask_all]

        norm_pos = np.linalg.norm(pos, axis=1)
        norm_neg = np.linalg.norm(neg, axis=1)
        pos_clean = pos[norm_pos < NORM_OUTLIER_THRESHOLD]
        neg_clean = neg[norm_neg < NORM_OUTLIER_THRESHOLD]

        np.savez(os.path.join(FEATURES_DIR, f'concept{cid}_clean.npz'), pos=pos_clean, neg=neg_clean)

        n_max = min(len(pos_clean), len(neg_clean)) // 2
        df = split_half_sweep(pos_clean, neg_clean, n_max=n_max, n_trials=200)
        df.to_csv(os.path.join(RESULTS_DIR, f'dim_angle_axbench_concept{cid}_clean_splithalf.csv'), index=False)
        C, r2 = fit_tan2_c(df['n'].values, df['mean'].values)

        mu, w_pos, w_neg, plain_mu = reweighted_dim_unpaired(pos_clean, neg_clean)
        angle_rw = np.degrees(np.arccos(np.clip(mu @ plain_mu, -1, 1)))
        n_eff_pos = 1.0 / np.sum(w_pos ** 2)
        n_eff_neg = 1.0 / np.sum(w_neg ** 2)

        np.savez(os.path.join(RESULTS_DIR, f'reweighted_unpaired_axbench_concept{cid}_clean.npz'),
                 mu=mu, plain_mu=plain_mu, w_pos=w_pos, w_neg=w_neg, pos=pos_clean, neg=neg_clean)

        row = {
            'concept_id': cid,
            'desc': concept_desc[cid],
            'n_pos': len(pos_clean),
            'n_neg': len(neg_clean),
            'C': C,
            'R2': r2,
            'angle_plain_vs_reweighted': angle_rw,
            'n_eff_pos': n_eff_pos,
            'n_eff_neg': n_eff_neg,
        }
        summary_rows.append(row)
        flag = ' <-- heterogeneous' if r2 < 0.9 else ''
        print(f"concept {cid:4d} ({concept_desc[cid]!r:45s}): C={C:7.2f} R2={r2:6.3f} "
              f"angle_rw={angle_rw:6.2f} N_eff=({n_eff_pos:.0f},{n_eff_neg:.0f}){flag}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS_DIR, 'axbench_all_concepts_summary.csv'), index=False)
    print(f"\nSaved summary to {os.path.join(RESULTS_DIR, 'axbench_all_concepts_summary.csv')}")
    print(f"\nHeterogeneous concepts (R2<0.9): {summary[summary.R2 < 0.9].concept_id.tolist()}")


if __name__ == "__main__":
    main()
