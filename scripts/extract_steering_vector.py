"""Extract a Theory-of-Mind steering vector via Contrastive Activation Addition (CAA).

For each game observation we run two forward passes:
  - base:    system + user prompt (no ToM cue)
  - steered: system + user prompt + tom_suffix

We hook the residual stream at a chosen layer, capture the last-token hidden
state, and accumulate:

    vector = mean(h_steered) - mean(h_base)

The resulting .npy file can be used directly with ActivationSteering.

Usage
-----
python scripts/extract_steering_vector.py \
    --model Qwen/Qwen3-4B \
    --layer model.layers.18 \
    --game beauty_contest \
    --num-samples 64 \
    --output vectors/tom_beauty_contest_l18.npy

Then in run_config.yaml:
    steering:
      default: activation
      per_agent:
        player_0:
          layer: model.layers.18
          vector_path: vectors/tom_beauty_contest_l18.npy
          coefficient: 20.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.registry import build_game  # noqa: E402

DEFAULT_TOM_SUFFIX = (
      """Assume the other players are rational and are also attempting to model your reasoning.

      Ask:

      * What contribution would another player expect me to choose?
      * What contribution would they choose in response to that expectation?
      * What contribution is most likely to emerge if all players reason in this way?

      Select the contribution that is most likely to be part of a mutually consistent set of choices across all players."""

)

# ── prompt generation ──────────────────────────────────────────────────────────

def _beauty_contest_prompts(renderer, num_samples: int):
    """Generate diverse (system, user) pairs for beauty_contest."""
    import random
    pairs = []
    for i in range(num_samples):
        n_players = random.choice([3, 4, 5, 6])
        round_idx = random.randint(0, 4)
        history = []
        for r in range(round_idx):
            guesses = {f"player_{p}": random.randint(0, 100) for p in range(n_players)}
            mean = sum(guesses.values()) / n_players
            target = mean * 2 / 3
            best = min(abs(v - target) for v in guesses.values())
            winners = [p for p, v in guesses.items() if abs(v - target) == best]
            history.append({
                "mean": round(mean, 1), "target": round(target, 1),
                "choices": guesses, "winners": winners,
            })
        obs = {
            "round_index": round_idx,
            "num_players": n_players,
            "history": history,
            "low": 0,
            "high": 100,
        }
        agent_id = f"player_{random.randint(0, n_players - 1)}"
        system = renderer.system_prompt(agent_id)
        user = renderer.render(obs, agent_id, None)
        pairs.append((system, user))
    return pairs


def _gbs_prompts(renderer, num_samples: int):
    """Generate diverse (system, user) pairs for GBS."""
    import random
    pairs = []
    for i in range(num_samples):
        n_players = random.choice([3, 4, 5])
        round_idx = random.randint(0, 5)
        history = []
        target = random.randint(1, 100)
        low, high = 1, 100
        for r in range(round_idx):
            guesses = {f"player_{p}": random.randint(low, high) for p in range(n_players)}
            median = sorted(guesses.values())[len(guesses) // 2]
            direction = "higher" if target > median else ("lower" if target < median else "correct")
            history.append({"median": median, "direction": direction})
            if direction == "higher":
                low = median + 1
            elif direction == "lower":
                high = median - 1
        obs = {
            "round_index": round_idx,
            "num_players": n_players,
            "history": history,
            "low": low,
            "high": high,
        }
        agent_id = f"player_{random.randint(0, n_players - 1)}"
        system = renderer.system_prompt(agent_id)
        user = renderer.render(obs, agent_id, None)
        pairs.append((system, user))
    return pairs


def generate_prompts(game: str, num_samples: int):
    if game == "beauty_contest":
        _, renderer, _ = build_game(family="symbolic", game_id="beauty_contest",
                                    num_players=4, env_kwargs={})
        return _beauty_contest_prompts(renderer, num_samples)
    if game == "gbs":
        _, renderer, _ = build_game(family="symbolic", game_id="gbs",
                                    num_players=4, env_kwargs={})
        return _gbs_prompts(renderer, num_samples)
    raise ValueError(f"Unknown game for prompt generation: {game}. "
                     "Add a generator or pass --prompt-file.")


# ── activation extraction ──────────────────────────────────────────────────────

def _resolve_submodule(model, dotted_name: str):
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def _last_token_hidden(model, tokenizer, system: str, user: str,
                       layer_module, device: str) -> torch.Tensor:
    """Return the last-token residual-stream vector at layer_module."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    captured = {}

    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: [batch, seq_len, hidden_dim] — take last token
        captured["h"] = h[0, -1, :].detach().float()

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return captured["h"]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--layer", default="model.layers.18",
                        help="Dotted submodule path to hook (default: model.layers.18)")
    parser.add_argument("--game", default="beauty_contest",
                        choices=["beauty_contest", "gbs"])
    parser.add_argument("--num-samples", type=int, default=64,
                        help="Number of contrastive prompt pairs (default: 64)")
    parser.add_argument("--suffix", default=DEFAULT_TOM_SUFFIX,
                        help="ToM suffix appended to user prompt for steered pass")
    parser.add_argument("--output", default=None,
                        help="Output .npy path (default: vectors/tom_<game>_<layer>.npy)")
    args = parser.parse_args()

    if args.output is None:
        layer_tag = args.layer.replace(".", "_")
        args.output = f"vectors/tom_{args.game}_{layer_tag}.npy"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {args.model} …")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device)
    model.eval()

    layer_module = _resolve_submodule(model, args.layer)
    print(f"Hooked layer: {args.layer}")
    print(f"Generating {args.num_samples} prompt pairs for game '{args.game}' …")

    pairs = generate_prompts(args.game, args.num_samples)

    diffs = []
    for i, (system, user) in enumerate(pairs):
        h_base = _last_token_hidden(model, tokenizer, system, user,
                                    layer_module, device)
        h_steered = _last_token_hidden(model, tokenizer, system,
                                       user + args.suffix, layer_module, device)
        diffs.append(h_steered - h_base)
        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/{args.num_samples} pairs processed")

    vector = torch.stack(diffs).mean(dim=0)  # [hidden_dim]
    print(f"Vector shape: {vector.shape}, norm: {vector.norm().item():.4f}")

    np.save(args.output, vector.cpu().numpy())
    print(f"Saved: {args.output}")
    print()
    print("To apply in run_config.yaml:")
    print(f"  steering:")
    print(f"    default: activation")
    print(f"    per_agent:")
    print(f"      player_0:")
    print(f"        layer: {args.layer}")
    print(f"        vector_path: {args.output}")
    print(f"        coefficient: 20.0  # tune this")


if __name__ == "__main__":
    main()
