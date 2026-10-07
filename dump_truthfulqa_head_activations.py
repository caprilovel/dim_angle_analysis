# Export TruthfulQA per-head activations (true/false statements) at the
# cached top head for a given model, as a plain {pos, neg} npz -- matching
# the format used by reweighted_dim_unpaired.py and the refusal/sycophancy
# plotting scripts. pos = true statements, neg = false statements.

import os
import sys
import json
import argparse
import numpy as np
from einops import rearrange

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_HONEST_LLAMA_DIR = os.path.normpath(os.path.join(_HERE, '..', 'honest_llama'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_head_cache', type=str, default='results/top_head_cache.json')
    parser.add_argument('--honest_llama_dir', type=str, default=_DEFAULT_HONEST_LLAMA_DIR)
    parser.add_argument('--out_dir', type=str, default=os.path.join(_HERE, 'features_truthfulqa'))
    args = parser.parse_args()

    sys.path.append(args.honest_llama_dir)
    from utils import get_separated_activations

    os.makedirs(args.out_dir, exist_ok=True)

    features_dir = os.path.join(args.honest_llama_dir, 'features')
    head_wise_activations = np.load(f"{features_dir}/{args.model_name}_{args.dataset_name}_head_wise.npy")
    labels = np.load(f"{features_dir}/{args.model_name}_{args.dataset_name}_labels.npy")

    n_statements, num_layers, hidden_size = head_wise_activations.shape

    from transformers import AutoConfig
    from constants import HF_NAMES
    hf_config = AutoConfig.from_pretrained(HF_NAMES[args.model_name])
    num_heads = hf_config.num_attention_heads
    head_dim = hidden_size // num_heads

    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)
    separated_head_wise_activations, separated_labels, _ = get_separated_activations(labels, head_wise_activations)
    n_questions = len(separated_labels)

    cache_key = f"{args.model_name}_{args.dataset_name}_seed{args.seed}_val{args.val_ratio}"
    with open(args.top_head_cache) as f:
        cache_data = json.load(f)
    cached = cache_data[cache_key]
    layer, head = cached['layer'], cached['head']
    print(f"Using cached top head: layer={layer}, head={head}")

    acts_all = np.concatenate(
        [separated_head_wise_activations[i][:, layer, head, :] for i in range(n_questions)], axis=0
    )
    labs_all = np.concatenate([separated_labels[i] for i in range(n_questions)], axis=0)

    pos_acts = acts_all[labs_all == 1]  # true statements
    neg_acts = acts_all[labs_all == 0]  # false statements
    print(f"pos (true): {pos_acts.shape}, neg (false): {neg_acts.shape}")

    out_path = os.path.join(args.out_dir, f"{args.model_name}_tqa_layer{layer}_head{head}.npz")
    np.savez(out_path, pos=pos_acts, neg=neg_acts)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
