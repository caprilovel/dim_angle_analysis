# CAA-style sycophancy activation dump, mirroring dump_refusal_activations.py:
# for each contrastive pair, build pos_prompt (ends with the sycophantic
# answer) and neg_prompt (ends with the non-sycophantic answer), run the
# model, and capture the residual-stream activation at the actual answer
# token position (not necessarily the prompt's last token -- see
# data.answer_letter_token_index), for a chosen layer.

import os
import sys
import argparse
import torch
import numpy as np

from common import ANTHROPIC_AI_RISK_DIR
sys.path.insert(0, ANTHROPIC_AI_RISK_DIR)

from data import make_contrastive_pairs, pair_to_prompts, answer_letter_token_index
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_capture_hook(store, token_idx_per_sample, sample_counter):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.detach().to(torch.float32).cpu()
        for b in range(hidden.shape[0]):
            idx = token_idx_per_sample[sample_counter[0] + b]
            store.append(hidden[b, idx, :].clone())
        sample_counter[0] += hidden.shape[0]
    return hook_fn


@torch.no_grad()
def extract_batch(model, tokenizer, prompts, layer, batch_size=8, max_length=512):
    block = model.model.layers[layer]
    acts = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        token_idxs = [answer_letter_token_index(p, tokenizer) for p in batch]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length,
                         add_special_tokens=False)
        # left-pad aware: recompute token index relative to padded sequence
        input_ids = enc['input_ids']
        attn = enc['attention_mask']
        pad_side = tokenizer.padding_side
        seq_lens = attn.sum(dim=1).tolist()

        store = []
        counter = [0]
        adj_idxs = {}
        for b, (tidx, slen) in enumerate(zip(token_idxs, seq_lens)):
            total_len = input_ids.shape[1]
            n_pad = total_len - slen
            if pad_side == 'left':
                adj_idxs[b] = n_pad + tidx
            else:
                adj_idxs[b] = tidx

        def hook_fn(module, inp, outp):
            hidden = outp[0] if isinstance(outp, tuple) else outp
            hidden = hidden.detach().to(torch.float32).cpu()
            for b in range(hidden.shape[0]):
                store.append(hidden[b, adj_idxs[b], :].clone())

        handle = block.register_forward_hook(hook_fn)
        try:
            model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device))
        finally:
            handle.remove()

        acts.extend(store)

    return torch.stack(acts, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_tag', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--dataset', type=str, default='sycophancy_on_political_typology_quiz')
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--out_dir', type=str,
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features_sycophancy'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto',
                                                  torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()

    pairs = make_contrastive_pairs(args.dataset, n_samples=args.n_samples, seed=args.seed)
    print(f"Loaded {len(pairs)} contrastive pairs from {args.dataset}")

    pos_prompts, neg_prompts = [], []
    for pair in pairs:
        pos_p, neg_p = pair_to_prompts(pair, tokenizer)
        pos_prompts.append(pos_p)
        neg_prompts.append(neg_p)

    print("Extracting sycophantic (pos) activations...")
    pos_acts = extract_batch(model, tokenizer, pos_prompts, args.layer, batch_size=args.batch_size)
    print("pos_acts:", pos_acts.shape)

    print("Extracting non-sycophantic (neg) activations...")
    neg_acts = extract_batch(model, tokenizer, neg_prompts, args.layer, batch_size=args.batch_size)
    print("neg_acts:", neg_acts.shape)

    out_path = os.path.join(args.out_dir, f"{args.model_tag}_sycophancy_layer{args.layer}.npz")
    np.savez(out_path, pos=pos_acts, neg=neg_acts)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
