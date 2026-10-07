# Small helpers shared across the dump/sweep/fit/reweight/plot scripts in
# this repo. Kept dependency-free (numpy only) so every script can import
# it without pulling in matplotlib or the external pipeline repos.

import numpy as np


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
