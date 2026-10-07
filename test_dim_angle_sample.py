# Test: how much does the DiM (difference-in-means) direction rotate
# when estimated from only n positive + n negative statements,
# compared to the direction estimated from the full training set?
#
# Steps:
#   1. Find the best (layer, head) to steer, using the same probe-accuracy
#      selection as get_top_heads() (top-1 head). Cached to disk since
#      training 1024 logistic-regression probes is slow.
#   2. At that (layer, head), compute the "full" DiM direction from all
#      available true/false activations.
#   3. For each sample size n in a given range, repeatedly subsample
#      n true + n false statements (no replacement within each draw),
#      recompute DiM direction, and measure the angle (in degrees) to the
#      full-data direction.
#   4. Report min/max/mean angle vs. n, and dump the raw table to CSV.
#
# This script depends on honest_llama's utils.py/constants.py (TruthfulQA
# per-head activation pipeline) and its features/ directory for cached
# activations -- it is not fully standalone. Point --honest_llama_dir at
# that repo if it is not a sibling of this directory.

import numpy as np
import argparse
import json
import os
from einops import rearrange

import sys

from common import angle_between

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_HONEST_LLAMA_DIR = os.path.normpath(os.path.join(_HERE, '..', 'honest_llama'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3_8B_instruct')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--mode', type=str, default='split_half', choices=['full', 'split_half'])
    parser.add_argument('--n_min', type=int, default=1)
    parser.add_argument('--n_max', type=int, default=100)
    parser.add_argument('--n_trials', type=int, default=200, help='random subsampling trials per sample size')
    parser.add_argument('--log_points', type=int, default=0,
                         help='if >0, use this many log-spaced n values from n_min to n_max instead of every integer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2, help='val split used only for picking the top head')
    parser.add_argument('--top_head_cache', type=str, default='results/top_head_cache.json')
    parser.add_argument('--out_csv', type=str, default='results/dim_angle_vs_n.csv')
    parser.add_argument('--honest_llama_dir', type=str, default=_DEFAULT_HONEST_LLAMA_DIR,
                         help='path to the honest_llama repo (for utils.py, constants.py, features/)')
    args = parser.parse_args()

    sys.path.append(args.honest_llama_dir)
    from utils import get_separated_activations, get_top_heads
    from constants import HF_NAMES

    rng = np.random.default_rng(args.seed)

    features_dir = os.path.join(args.honest_llama_dir, 'features')
    head_wise_activations = np.load(f"{features_dir}/{args.model_name}_{args.dataset_name}_head_wise.npy")
    labels = np.load(f"{features_dir}/{args.model_name}_{args.dataset_name}_labels.npy")

    n_statements, num_layers, hidden_size = head_wise_activations.shape

    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(HF_NAMES[args.model_name])
    num_heads = hf_config.num_attention_heads
    assert hidden_size % num_heads == 0, f"hidden_size={hidden_size} not divisible by num_heads={num_heads}"
    head_dim = hidden_size // num_heads

    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)
    separated_head_wise_activations, separated_labels, _ = get_separated_activations(labels, head_wise_activations)

    n_questions = len(separated_labels)
    all_idxs = np.arange(n_questions)

    cache_key = f"{args.model_name}_{args.dataset_name}_seed{args.seed}_val{args.val_ratio}"
    cached = None
    if os.path.exists(args.top_head_cache):
        with open(args.top_head_cache) as f:
            cache_data = json.load(f)
        cached = cache_data.get(cache_key)

    if cached is not None:
        layer, head = cached['layer'], cached['head']
        print(f"Using cached top head: layer={layer}, head={head}")
    else:
        rng_np = np.random.RandomState(args.seed)
        val_set_idxs = rng_np.choice(all_idxs, size=int(n_questions * args.val_ratio), replace=False)
        train_set_idxs = np.array([x for x in all_idxs if x not in set(val_set_idxs)])

        print(f"Selecting top head via probe accuracy (train={len(train_set_idxs)} q, val={len(val_set_idxs)} q)...")
        top_heads, _ = get_top_heads(train_set_idxs, val_set_idxs,
                                      separated_head_wise_activations, separated_labels,
                                      num_layers, num_heads, args.seed, num_to_intervene=1)
        layer, head = top_heads[0]
        print(f"Top head selected: layer={layer}, head={head}")

        os.makedirs(os.path.dirname(args.top_head_cache), exist_ok=True)
        cache_data = {}
        if os.path.exists(args.top_head_cache):
            with open(args.top_head_cache) as f:
                cache_data = json.load(f)
        cache_data[cache_key] = {'layer': int(layer), 'head': int(head)}
        with open(args.top_head_cache, 'w') as f:
            json.dump(cache_data, f, indent=2)

    acts_all = np.concatenate(
        [separated_head_wise_activations[i][:, layer, head, :] for i in range(n_questions)], axis=0
    )
    labs_all = np.concatenate([separated_labels[i] for i in range(n_questions)], axis=0)

    pos_acts = acts_all[labs_all == 1]
    neg_acts = acts_all[labs_all == 0]
    print(f"Total statements: {len(acts_all)}  (pos={len(pos_acts)}, neg={len(neg_acts)})")

    n_pos_max = len(pos_acts)
    n_neg_max = len(neg_acts)

    if args.mode == 'full':
        full_direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
        n_max_usable = min(args.n_max, n_pos_max, n_neg_max)
    else:
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
        print(f"n={n:5d}  min={rows[-1]['min']:.2f}  mean={rows[-1]['mean']:.2f}  max={rows[-1]['max']:.2f}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    import csv
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-n angle stats to {args.out_csv}")
    print(f"(mode={args.mode}, layer={layer}, head={head}, head_dim={head_dim}, n_trials={args.n_trials})")


if __name__ == "__main__":
    main()
