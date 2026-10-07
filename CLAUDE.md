# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of standalone analysis/plotting scripts (no package, no git repo,
no shared `main`) that study one question: when you estimate a
difference-in-means (DiM) direction `mu = normalize(mean(pos) - mean(neg))`
from `n` paired or unpaired samples of LLM residual-stream activations, how
much does that direction rotate away from the "true" population direction as
`n` shrinks, and how heterogeneous is the underlying pool?

Four concept datasets are analyzed in parallel, each extracted from a
different model/pipeline:
- **Refusal** (Arditi et al. harmful/harmless instructions) — `*_refusal*`
- **Sycophancy** (CAA-style contrastive pairs, Anthropic AI-risk eval) —
  `*_sycophancy*`
- **TruthfulQA** (per-head true/false statement activations, honest_llama
  ITI pipeline) — `*_truthfulqa*`
- **Axbench concepts** (20 concepts, gemma-2-9B layer 31, from a cached
  `quadratic_probe` tensor) — `*_axbench*`

## Pipeline shape (per dataset)

Each dataset follows the same four-stage flow, implemented as separate CLI
scripts chained by file paths rather than imports:

1. **Dump** (`dump_*.py`) — run the source model/pipeline once, extract raw
   per-sample activations at a chosen (layer, position), save as
   `features_<dataset>/*.npz` with `pos`/`neg` (or `harmful`/`harmless`)
   arrays of shape `(n_samples, hidden_size)`. These scripts reach into
   sibling repos not tracked here (see "External dependencies" below) and
   are not meant to be re-run casually — the `.npz` outputs are the
   reusable artifact.
2. **Angle sweep** (`test_dim_angle_generic.py`, `test_dim_angle_sample.py`,
   or the sweep embedded in `run_all_axbench_concepts.py`) — for a range of
   sample sizes `n`, repeatedly subsample and recompute a DiM direction,
   measure the angle to a reference direction, write per-`n` angle stats to
   `results/dim_angle_*.csv`. Use `--mode split_half` (two disjoint
   subsamples from the same pool) unless you specifically want `--mode
   full` (subsample vs. full-pool direction, which is biased as `n`
   approaches pool size — see the docstring in
   `test_dim_angle_generic.py`).
3. **Fit** (`fit_dim_angle_c.py`) — fit `tan^2(theta_single) ≈ C/n` to a
   split-half CSV to get an effective inverse-SNR `C` and an R² goodness-of-
   fit; low R² or a log-log slope far from -1 flags a heterogeneous pool
   (subgroup structure, not i.i.d. noise). This diagnostic is the core
   lens the whole repo is built around.
4. **Reweight** (`reweighted_dim.py` for paired `z_i = pos_i - neg_i`,
   `reweighted_dim_unpaired.py` for independent pos/neg groups) — a
   cross-fitted, iteratively-reweighted DiM estimator that downweights
   pairs/samples with high out-of-direction noise or disagreeing sign, to
   see whether heterogeneity can be corrected rather than just detected.

Plotting scripts (`plot_*.py`) are one-off, figure-specific consumers of the
`results/*.csv` and `features_*/*.npz` / `results/reweighted_*.npz` outputs —
they are not a shared plotting library, each hardcodes its own input paths
and output filename under `figures/`.

`run_all_axbench_concepts.py` is the exception: it inlines stages 2–4
(split-half sweep, C/R² fit, unpaired reweighting) for all 20 axbench
concepts in one script and writes `results/axbench_all_concepts_summary.csv`.

`stratify_by_letter.py` is a one-off diagnostic script, not part of the
main pipeline: it re-slices already-extracted sycophancy activations by the
answer-letter nuisance variable to confirm a specific confound hypothesis.

## Running scripts

No build/lint/test tooling exists (no `requirements.txt`, no test runner) —
this is exploratory research code, each script is run directly, e.g.:

```bash
python test_dim_angle_generic.py --npz_path features_refusal/llama3_8B_instruct_refusal_layer12_pos-5.npz \
    --mode split_half --out_csv results/dim_angle_refusal_llama3.csv --tag "Refusal / llama3"

python fit_dim_angle_c.py --csv results/dim_angle_refusal_llama3.csv --tag "Refusal / llama3"

python run_all_axbench_concepts.py
```

Every script takes `argparse` CLI flags — check `--help` or the top of
`main()` rather than guessing flag names. Random seeds default to 42
throughout for reproducibility of the subsampling sweeps.

## External dependencies (not in this repo)

The `dump_*.py` scripts and a couple of others (`stratify_by_letter.py`,
`test_dim_angle_sample.py`) import from sibling repos by absolute path or
`sys.path.insert`, not pip packages:
- `/home/zhu.3723/code/code/quadratic_probe/quad_refusal/refusal-aird` —
  Arditi et al. refusal-direction pipeline (`pipeline.model_utils`,
  `pipeline.utils.hook_utils`, `dataset.load_dataset`)
- `/home/zhu.3723/code/code/quadratic_probe/quad_exp/anthropic_ai_risk` —
  sycophancy contrastive-pair construction (`data.make_contrastive_pairs`,
  `data.pair_to_prompts`, `data.answer_letter_token_index`)
- `honest_llama` (expected as a sibling directory, or pass
  `--honest_llama_dir`) — TruthfulQA per-head activation utilities
  (`utils.get_separated_activations`)
- `/research/nfs_khalili_17/ding/code/quadratic_probe/qb/cache/expanded_9b_l31/hidden_states_axbench.pt`
  — cached axbench hidden states tensor, read directly by
  `dump_axbench_concept.py` and `run_all_axbench_concepts.py`

If one of these paths has moved, the dump script's import will fail —
that's the first thing to check before debugging further.

## Plot styling conventions

All plotting scripts share the same conventions, worth matching in any new
plot:
- `matplotlib.use('Agg')` + `rcParams['pdf.fonttype'] = rcParams['ps.fonttype'] = 42`
  (embeds editable text in PDF/PS output instead of rasterized Type 3 fonts)
- A fixed two-color palette for pos/neg or ok/flagged status:
  `COLOR_POS = '#2a78d6'` (blue), `COLOR_NEG`/`COLOR_FLAG = '#eb6834'`
  (orange) — used for binary group/status distinctions, not arbitrary
  categorical series, and labeled directly (title/annotation) rather than
  always via a legend.
- Figures are saved under `figures/`, raw per-run numeric output under
  `results/`, extracted activations under `features_<dataset>/`.
