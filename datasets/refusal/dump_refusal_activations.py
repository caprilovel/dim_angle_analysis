# Dump raw per-instruction activations (not just the streaming mean) for the
# refusal direction setup (Arditi et al.), at a chosen (position, layer),
# for harmful vs harmless instructions. Reuses pipeline/model_utils + hook
# utilities from the quad_refusal/refusal-aird copy without modifying it.

import os
import sys
import json
import argparse
import torch
import numpy as np

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, 'features_refusal')

sys.path.insert(0, _REPO_ROOT)
from common import REFUSAL_AIRD_DIR
sys.path.insert(0, REFUSAL_AIRD_DIR)
os.chdir(REFUSAL_AIRD_DIR)

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks
from dataset.load_dataset import load_dataset_split


def get_capture_pre_hook(layer, store, position):
    def hook_fn(module, input):
        activation = input[0].detach().to(torch.float32).cpu()
        store[layer].append(activation[:, position, :].clone())
    return hook_fn


@torch.no_grad()
def dump_activations(model_base, instructions, layer, position, batch_size=32):
    n_layers = model_base.model.config.num_hidden_layers
    store = {layer: []}

    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_capture_pre_hook(layer, store, position))]

    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model_base.model(
                input_ids=inputs.input_ids.to(model_base.model.device),
                attention_mask=inputs.attention_mask.to(model_base.model.device),
            )

    return torch.cat(store[layer], dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_tag', type=str, required=True, help='short name for output file')
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--pos', type=int, required=True, help='negative index into eoi positions, e.g. -1 or -5')
    parser.add_argument('--n_harmless', type=int, default=None, help='subsample harmless_train to this many (default: all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', type=str, default=_DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model_base = construct_model_base(args.model_path)

    harmful = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)

    rng = np.random.RandomState(args.seed)
    if args.n_harmless is not None and args.n_harmless < len(harmless):
        idx = rng.choice(len(harmless), size=args.n_harmless, replace=False)
        harmless = [harmless[i] for i in idx]

    print(f"harmful: {len(harmful)}, harmless: {len(harmless)}")

    print("Extracting harmful activations...")
    harmful_acts = dump_activations(model_base, harmful, args.layer, args.pos, batch_size=args.batch_size)
    print("harmful_acts:", harmful_acts.shape)

    print("Extracting harmless activations...")
    harmless_acts = dump_activations(model_base, harmless, args.layer, args.pos, batch_size=args.batch_size)
    print("harmless_acts:", harmless_acts.shape)

    out_path = os.path.join(args.out_dir, f"{args.model_tag}_refusal_layer{args.layer}_pos{args.pos}.npz")
    np.savez(out_path, harmful=harmful_acts, harmless=harmless_acts)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
