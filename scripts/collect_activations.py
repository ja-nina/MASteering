"""Collect residual-stream activations from real multi-round game traces.

Runs game episodes with a random policy to build realistic multi-round
histories, then passes each agent's prompt through the model (with and
without the ToM suffix) and captures activations at a chosen layer.

Two output files per run:
  activations/base_{game}_l{N}.npy   — [T, d_model]  all token positions,
                                        base prompts (used to train SAE)
  activations/tom_{game}_l{N}.npy    — [P, d_model]  last token only,
                                        ToM-suffix prompts (used to find
                                        differential features)
  activations/base_last_{game}_l{N}.npy — [P, d_model] last token of base
                                          (paired with tom for differential)

Usage
-----
python scripts/collect_activations.py \\
    --game beauty_contest \\
    --num-episodes 200 \\
    --max-rounds 4 \\
    --layer model.layers.18 \\
    --model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.registry import build_game  # noqa: E402

TOM_SUFFIX_BEAUTY_CONTEST = (
    "\n\nBefore you answer, think carefully about what the other players "
    "are likely to do. Model their reasoning and best-respond to it."
)

TOM_SUFFIX_GBS = (
    "\n\nBefore you answer, think carefully about what the other players "
    "are likely to contribute. Coordinate with them so your contributions "
    "sum to the target as quickly as possible."
)

TOM_SUFFIX_BY_GAME = {
    "beauty_contest": TOM_SUFFIX_BEAUTY_CONTEST,
    "gbs": TOM_SUFFIX_GBS,
}


# ── random policy ─────────────────────────────────────────────────────────────

class _RandomPolicy:
    """Generates random valid actions without loading any model."""

    def act(self, system: str, user: str, agent_id: str, steering) -> str:
        # Goldstone GBS: submit a random non-negative contribution
        if "CONTRIBUTION:" in user:
            return f"CONTRIBUTION: {random.randint(0, 50)}"
        # beauty_contest: parse allowed range from the rendered user prompt
        low, high = 0, 100
        for line in user.splitlines():
            if "between" in line and "and" in line:
                parts = line.replace("(inclusive)", "").split()
                try:
                    idx = parts.index("between")
                    low = int(parts[idx + 1].strip(".,"))
                    high = int(parts[idx + 3].strip(".,"))
                except (ValueError, IndexError):
                    pass
                break
        return f"CHOICE: {random.randint(low, high)}"


# ── episode runner (no model) ─────────────────────────────────────────────────

def run_episode_collect(env, renderer, parser, policy, max_rounds: int):
    """Run one episode and return list of (system, user) prompt pairs."""
    env.reset()
    prompts = []
    done = False
    round_idx = 0

    while not done and round_idx < max_rounds:
        pending = env.pending()
        actions = {}
        for agent_id, raw_obs in pending:
            system = renderer.system_prompt(agent_id)
            user = renderer.render(raw_obs, agent_id, None)
            prompts.append((system, user))
            completion = policy.act(system, user, agent_id, None)
            result = parser.parse(completion, raw_obs, agent_id, None)
            from testbed.types import ParsedAction
            actions[agent_id] = result.value if isinstance(result, ParsedAction) else 0
        result = env.submit(actions)
        done = result.done
        round_idx += 1

    return prompts


# ── activation extraction ─────────────────────────────────────────────────────

def _resolve_submodule(model, dotted_name: str):
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def _extract_all_tokens(model, tokenizer, system: str, user: str,
                        layer_module, device: str) -> torch.Tensor:
    """Return all token hidden states at layer: [seq_len, d_model]."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    captured = {}

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[0].detach().float()  # [seq_len, d_model]

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured["h"]


def _extract_last_token(model, tokenizer, system: str, user: str,
                        layer_module, device: str) -> torch.Tensor:
    """Return last-token hidden state: [d_model]."""
    return _extract_all_tokens(
        model, tokenizer, system, user, layer_module, device)[-1]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="beauty_contest",
                    choices=["beauty_contest", "gbs"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--layer", default="model.layers.18")
    ap.add_argument("--num-episodes", type=int, default=500)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--output-dir", default="/scratch/inf0/user/nzukowsk/activations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_tag = args.layer.replace(".", "_")
    out_combined  = os.path.join(args.output_dir,
                                 f"base_{args.game}_{layer_tag}.npy")  # base_ prefix for train_sae.py
    out_base_last = os.path.join(args.output_dir,
                                 f"base_last_{args.game}_{layer_tag}.npy")
    out_tom_last  = os.path.join(args.output_dir,
                                 f"tom_{args.game}_{layer_tag}.npy")

    # ── 1. generate game traces (no model needed) ────────────────────────────
    print(f"Generating {args.num_episodes} episodes "
          f"(game={args.game}, players={args.num_players}, "
          f"max_rounds={args.max_rounds}) ...")
    policy = _RandomPolicy()
    all_prompts: list[tuple[str, str]] = []

    for ep in range(args.num_episodes):
        env, renderer, parser = build_game(
            family="symbolic", game_id=args.game,
            num_players=args.num_players, env_kwargs={})
        prompts = run_episode_collect(
            env, renderer, parser, policy, args.max_rounds)
        all_prompts.extend(prompts)
        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{args.num_episodes} episodes "
                  f"({len(all_prompts)} prompts so far)")

    print(f"Total prompts: {len(all_prompts)}")

    # ── 2. load model ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Loading {args.model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device)
    model.eval()
    layer_module = _resolve_submodule(model, args.layer)
    print(f"Hooked: {args.layer}")

    # ── 3. extract activations (stream to disk to avoid OOM) ────────────────
    tom_suffix = TOM_SUFFIX_BY_GAME[args.game]

    base_last_list = []
    tom_last_list  = []

    # open memmap files for all-token activations — written row by row so
    # we never hold more than one prompt's activations in RAM at once
    sample_base = _extract_all_tokens(model, tokenizer,
                                      all_prompts[0][0], all_prompts[0][1],
                                      layer_module, device)
    d_model = sample_base.shape[1]

    # rough upper bound on total tokens (seq_len varies; 300 is conservative)
    max_tokens = len(all_prompts) * 300 * 2   # ×2 for base + ToM

    combined_mm = np.lib.format.open_memmap(
        out_combined, mode="w+", dtype="float32",
        shape=(max_tokens, d_model))
    row = 0

    for i, (system, user) in enumerate(all_prompts):
        h_base_all = _extract_all_tokens(model, tokenizer, system, user,
                                         layer_module, device)
        h_tom_all  = _extract_all_tokens(model, tokenizer, system,
                                         user + tom_suffix, layer_module, device)

        # stream to memmap
        for h in (h_base_all, h_tom_all):
            t = h.shape[0]
            combined_mm[row:row + t] = (h.cpu() if h.is_cuda else h).numpy()
            row += t

        base_last_list.append(h_base_all[-1].cpu())
        tom_last_list.append(h_tom_all[-1].cpu())

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_prompts)} prompts processed  "
                  f"({row} activation rows written)", flush=True)

    # truncate memmap to actual number of rows written
    combined_mm.flush()
    del combined_mm
    # reload, truncate, re-save as a proper npy
    data = np.load(out_combined, mmap_mode="r")[:row]
    np.save(out_combined, data)
    del data

    # ── 4. save paired last-token arrays ────────────────────────────────────
    base_last = torch.stack(base_last_list).numpy()
    tom_last  = torch.stack(tom_last_list).numpy()
    np.save(out_base_last, base_last)
    np.save(out_tom_last,  tom_last)

    print(f"\nSaved:", flush=True)
    print(f"  {out_combined}   [{row}, {d_model}]  (SAE training: base + ToM tokens)")
    print(f"  {out_base_last}  {base_last.shape}  (paired base last-token)")
    print(f"  {out_tom_last}   {tom_last.shape}  (paired ToM last-token)")


if __name__ == "__main__":
    main()
