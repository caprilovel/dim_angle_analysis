# Export a single axbench concept's pos/neg activations as a plain
# {pos, neg} npz, matching the format used by the refusal/sycophancy
# scripts. Source: qb/cache/expanded_9b_l31/hidden_states_axbench.pt
# (gemma-9b, layer 31; 20 concepts, 300 pos per concept, 300 shared neg).

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from common import AXBENCH_CACHE_PATH


def load_axbench_cache(cache_path):
    """Load the cached axbench hidden-states dict and return (hidden,
    labels, concept_ids, prompts) as plain numpy arrays."""
    d = torch.load(cache_path, map_location='cpu', weights_only=False)
    layer_key = [k for k in d.keys() if isinstance(k, int)][0]
    hidden = d[layer_key].numpy()
    labels = d['labels'].numpy() if torch.is_tensor(d['labels']) else np.asarray(d['labels'])
    concept_ids = np.asarray(d['concept_ids'])
    prompts = np.asarray(d['prompts'], dtype=object)
    return hidden, labels, concept_ids, prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', type=str, default=AXBENCH_CACHE_PATH)
    parser.add_argument('--concept_id', type=int, required=True)
    parser.add_argument('--out_npz', type=str, required=True)
    args = parser.parse_args()

    hidden, labels, concept_ids, _ = load_axbench_cache(args.cache_path)

    pos_mask = (labels == 1) & (concept_ids == args.concept_id)
    neg_mask = (labels == 0)

    pos = hidden[pos_mask]
    neg = hidden[neg_mask]
    print(f"concept_id={args.concept_id}: pos={pos.shape}, neg={neg.shape}")

    np.savez(args.out_npz, pos=pos, neg=neg)
    print(f"Saved to {args.out_npz}")


if __name__ == "__main__":
    main()
