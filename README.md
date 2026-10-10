# dim_angle_analysis

A collection of standalone analysis/plotting scripts that study one
question: when you estimate a difference-in-means (DiM) direction
`mu = normalize(mean(pos) - mean(neg))` from `n` paired or unpaired
samples of LLM residual-stream activations, how much does that direction
rotate away from the "true" population direction as `n` shrinks, and how
heterogeneous is the underlying pool?

Four concept datasets are analyzed in parallel, each extracted from a
different model/pipeline:

- **Refusal** (Arditi et al. harmful/harmless instructions) — `*_refusal*`
- **Sycophancy** (CAA-style contrastive pairs, Anthropic AI-risk eval) —
  `*_sycophancy*`
- **TruthfulQA** (per-head true/false statement activations, honest_llama
  ITI pipeline) — `*_truthfulqa*`
- **Axbench concepts** (20 concepts, gemma-2-9B layer 31, from a cached
  `quadratic_probe` tensor) — `*_axbench*`

> See `CLAUDE.md` for the version of this guide aimed at Claude Code /
> coding agents (directory layout, external-dependency paths, plot
> styling conventions). This file is the human-facing overview.

## Pipeline shape (per dataset)

Each dataset follows the same flow, implemented as separate CLI scripts
chained by file paths rather than imports:

1. **Dump** (`datasets/<name>/dump_*.py`) — run the source model/pipeline
   once, extract raw per-sample activations at a chosen (layer, position),
   save as `features_<dataset>/*.npz` with `pos`/`neg` (or
   `harmful`/`harmless`) arrays of shape `(n_samples, hidden_size)`.
2. **Angle sweep** (`test_dim_angle_generic.py`, or the dataset-specific
   drivers under `datasets/truthfulqa/` and `datasets/axbench/`) — for a
   range of sample sizes `n`, repeatedly subsample and recompute a DiM
   direction, measure the angle to a reference direction, write per-`n`
   angle stats to `results/dim_angle_*.csv`.
3. **Fit** (`fit_dim_angle_c.py`, or `common.fit_tan2_c`) — fit
   `tan^2(theta_single) ≈ C/n` to a split-half CSV to get an effective
   inverse-SNR `C` and an R² goodness-of-fit; low R² or a log-log slope
   far from -1 flags a heterogeneous pool (subgroup structure, not i.i.d.
   noise). This diagnostic is the core lens the whole repo is built
   around.
4. **Robust re-estimate** — given the heterogeneity the fit step flags,
   try to recover a better direction than the plain sample mean. Three
   estimators are available, all consumable by the same comparison plots:
   - `reweighted_dim.py` — iteratively reweighted DiM for **paired**
     differences `z_i = pos_i - neg_i`; downweights pairs that disagree
     in sign with the current direction or have high out-of-direction
     noise.
   - `reweighted_dim_unpaired.py` — same idea for **independent** pos/neg
     groups (no sign gating, only noise-based downweighting — see its
     docstring for why sign doesn't apply there).
   - `geometric_median_dim.py` — a different robust-mean construction:
     split each pool into `k_groups` disjoint groups, take each group's
     ordinary mean, then take the **geometric median** (Weiszfeld's
     algorithm) of those group means instead of a single pooled mean.
     Tolerates a bounded fraction of corrupted groups without the
     per-sample weighting machinery of the two scripts above. Supports
     both `--mode paired` and `--mode unpaired`.

Plotting scripts (`plot_*.py` at the root, or under `datasets/<name>/` if
dataset-specific) are one-off, figure-specific consumers of the
`results/*.csv` and `features_*/*.npz` / `results/*.npz` outputs.

## Running scripts

No build/lint/test tooling exists — this is exploratory research code,
run directly from the repo root, e.g.:

```bash
# 1. angle sweep
python test_dim_angle_generic.py --npz_path features_refusal/llama3_8B_instruct_refusal_layer12_pos-5.npz \
    --mode split_half --out_csv results/dim_angle_refusal_llama3.csv --tag "Refusal / llama3"

# 2. fit C / R^2
python fit_dim_angle_c.py --csv results/dim_angle_refusal_llama3.csv --tag "Refusal / llama3"

# 3a. robust re-estimate: iteratively-reweighted DiM (unpaired groups)
python reweighted_dim_unpaired.py --npz_path features_refusal/llama3_8B_instruct_refusal_layer12_pos-5.npz \
    --pos_key harmful --neg_key harmless --out_npz results/reweighted_unpaired_llama3_refusal.npz

# 3b. robust re-estimate: geometric median of group means (unpaired groups)
python geometric_median_dim.py --npz_path features_refusal/llama3_8B_instruct_refusal_layer12_pos-5.npz \
    --mode unpaired --pos_key harmful --neg_key harmless --k_groups 9 \
    --out_npz results/gm_dim_refusal_llama3.npz

# 4. compare plain vs. robust direction (reads either reweighted_* or gm_dim_* npz,
#    both share the mu/plain_mu/pos/neg field names)
python plot_pos_neg_strength_compare_generic.py \
    --tag "Llama3-8B-Instruct" --npz results/gm_dim_refusal_llama3.npz \
    --pos_label harmful --neg_label harmless \
    --suptitle "Refusal: plain DiM vs geometric-median DiM" \
    --out_pdf figures/pos_neg_strength_refusal_gm_compare.pdf
```

Every script takes `argparse` CLI flags — check `--help` or the top of
`main()` rather than guessing flag names. Random seeds default to 42
throughout for reproducibility.

## Directory layout

```
common.py                                  shared helpers/constants/palette
test_dim_angle_generic.py                  angle sweep (stage 2)
fit_dim_angle_c.py                         C / R^2 fit (stage 3)
reweighted_dim.py                          robust re-estimate: paired, reweighting (stage 4)
reweighted_dim_unpaired.py                 robust re-estimate: unpaired, reweighting (stage 4)
geometric_median_dim.py                    robust re-estimate: group + geometric median (stage 4)
plot_pos_neg_strength_compare_generic.py   generic plain-vs-robust comparison plot
plot_pair_strength.py                      per-pair alignment/noise/weight plot (reweighted_dim.py output only)
plot_dim_angle_multi_model.py              cross-dataset summary figure
datasets/refusal/, datasets/sycophancy/,
datasets/truthfulqa/, datasets/axbench/    dataset-specific dump scripts, drivers, one-off figures
features_<dataset>/                        extracted activations (*.npz)
results/                                   sweep CSVs and reweighted/geometric-median *.npz outputs
figures/                                   output PDFs
```

## External dependencies (not in this repo)

The `datasets/<name>/dump_*.py` scripts and a couple of others import from
sibling repos by absolute path, not pip packages — see `CLAUDE.md` for the
exact paths and environment-variable overrides
(`DIM_ANGLE_ANTHROPIC_AI_RISK_DIR`, `DIM_ANGLE_REFUSAL_AIRD_DIR`,
`DIM_ANGLE_AXBENCH_CACHE_PATH`).
