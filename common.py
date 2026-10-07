# Small helpers shared across the dump/sweep/fit/reweight/plot scripts in
# this repo. Kept dependency-free (numpy only) so every script can import
# it without pulling in matplotlib or the external pipeline repos.

import os
import numpy as np

# Paths to sibling repos / cache files this repo's dump scripts depend on
# but does not itself contain (see CLAUDE.md "External dependencies").
# Override via environment variable if your checkout lives elsewhere.
ANTHROPIC_AI_RISK_DIR = os.environ.get(
    'DIM_ANGLE_ANTHROPIC_AI_RISK_DIR',
    '/home/zhu.3723/code/code/quadratic_probe/quad_exp/anthropic_ai_risk')
REFUSAL_AIRD_DIR = os.environ.get(
    'DIM_ANGLE_REFUSAL_AIRD_DIR',
    '/home/zhu.3723/code/code/quadratic_probe/quad_refusal/refusal-aird')
AXBENCH_CACHE_PATH = os.environ.get(
    'DIM_ANGLE_AXBENCH_CACHE_PATH',
    '/research/nfs_khalili_17/ding/code/quadratic_probe/qb/cache/expanded_9b_l31/hidden_states_axbench.pt')


def normalize(v):
    """Return v scaled to unit L2 norm."""
    return v / np.linalg.norm(v)


def angle_between(u, v):
    """Angle in degrees between two vectors."""
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def fold_assignment(n, K, rng):
    """Randomly assign n items to K folds; returns a length-n array of fold ids."""
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)
    fold_id = np.empty(n, dtype=int)
    for k, fold in enumerate(folds):
        fold_id[fold] = k
    return fold_id


def setup_paper_style():
    """Configure matplotlib for Agg backend + editable text in PDF/PS output.

    Call this before `import matplotlib.pyplot as plt` in any plotting script.
    """
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


# Shared two-color palette for pos/neg or ok/flagged binary distinctions.
# Not meant for arbitrary categorical series.
COLOR_POS = '#2a78d6'
COLOR_NEG = '#eb6834'
COLOR_OK = COLOR_POS
COLOR_FLAG = COLOR_NEG
