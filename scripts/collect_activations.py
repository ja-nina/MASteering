"""Collect residual-stream and MLP activations from real multi-round game traces.

Runs game episodes with a random policy to build realistic multi-round
histories, then passes each agent's prompt through the model (with and
without the ToM suffix) and captures activations for one or more streams
(resid, mlp) at a chosen layer, in a single forward pass.

Per stream, three output files:
  activations/base_{game}_l{N}_{stream}.npy   — [T, d_model]  all token
                                        positions, base prompts (SAE training)
  activations/tom_{game}_l{N}_{stream}.npy    — [P, d_model]  last token only,
                                        ToM-suffix prompts (differential)
  activations/base_last_{game}_l{N}_{stream}.npy — [P, d_model] last token of
                                        base (paired with tom for differential)

Usage
-----
python scripts/collect_activations.py \\
    --game beauty_contest \\
    --num-episodes 200 \\
    --max-rounds 4 \\
    --layer model.layers.18 \\
    --streams resid,mlp \\
    --model Qwen/Qwen3-4B
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
    """
      Assume the other players are rational and are also attempting to model your reasoning.

      Ask:

      * What contribution would another player expect me to choose?
      * What contribution would they choose in response to that expectation?
      * What contribution is most likely to emerge if all players reason in this way?

      Select the contribution that is most likely to be part of a mutually consistent set of choices across all players.
    """
)

TOM_SUFFIX_BY_GAME = {
    "beauty_contest": TOM_SUFFIX_BEAUTY_CONTEST,
    "gbs": TOM_SUFFIX_GBS,
}


# ── random policy ─────────────────────────────────────────────────────────────

class _RandomPolicy:
    """Generates random valid actions without loading any model."""

    def act(self, system: str, user: str, agent_id: str, steering) -> str:
        # Goldstone GBS: submit a random number
        if "NUMBER:" in user:
            return f"NUMBER: {random.randint(0, 50)}"
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


def _resolve_streams(model, layer: str, streams: list[str]) -> dict:
    """Map each requested stream name to the submodule to hook.

    resid -> the decoder block itself (post-residual-add output, i.e. the
             standard residual stream at this layer)
    mlp   -> the block's mlp submodule (its raw output, before being summed
             back into the residual stream)
    """
    modules = {}
    for stream in streams:
        if stream == "resid":
            modules[stream] = _resolve_submodule(model, layer)
        elif stream == "mlp":
            modules[stream] = _resolve_submodule(model, layer + ".mlp")
        else:
            raise ValueError(f"Unknown stream: {stream!r} (expected 'resid' or 'mlp')")
    return modules


def _extract_all_tokens(model, tokenizer, system: str, user: str,
                        layer_modules: dict, device: str) -> dict:
    """Return all token hidden states for each stream in one forward pass.

    layer_modules: {stream_name: submodule_to_hook}
    Returns: {stream_name: Tensor[seq_len, d_model]}
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    captured = {}

    def _make_hook(stream_name):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[stream_name] = h[0].detach().float()  # [seq_len, d_model]
        return hook

    handles = [module.register_forward_hook(_make_hook(name))
               for name, module in layer_modules.items()]
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return captured


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="beauty_contest",
                    choices=["beauty_contest", "gbs"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--layer", default="model.layers.18")
    ap.add_argument("--streams", default="resid,mlp",
                    help="Comma-separated stream names to capture: resid, mlp")
    ap.add_argument("--num-episodes", type=int, default=500)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--output-dir", default="/scratch/inf0/user/nzukowsk/activations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    streams = args.streams.split(",")

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_tag = args.layer.replace(".", "_")
    out_paths = {
        stream: {
            "combined": os.path.join(
                args.output_dir, f"base_{args.game}_{layer_tag}_{stream}.npy"),
            "base_last": os.path.join(
                args.output_dir, f"base_last_{args.game}_{layer_tag}_{stream}.npy"),
            "tom_last": os.path.join(
                args.output_dir, f"tom_{args.game}_{layer_tag}_{stream}.npy"),
        }
        for stream in streams
    }

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
    layer_modules = _resolve_streams(model, args.layer, streams)
    print(f"Hooked: {args.layer}  streams={streams}")

    # ── 3. count tokens first so memmaps are allocated at the exact right size ─
    tom_suffix = TOM_SUFFIX_BY_GAME[args.game]
    d_model    = model.config.hidden_size

    print("Counting tokens (no GPU)...", flush=True)
    total_tokens = 0
    for system, user in all_prompts:
        for text in (user, user + tom_suffix):
            msgs = [{"role": "system", "content": system},
                    {"role": "user",   "content": text}]
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            total_tokens += tokenizer(formatted, return_tensors="pt")["input_ids"].shape[1]
    print(f"  total tokens: {total_tokens}  "
          f"(~{total_tokens * d_model * 4 * len(streams) / 1e9:.1f} GB on disk "
          f"across {len(streams)} stream(s))", flush=True)

    # ── 4. allocate exact-size memmaps — no rewrite needed ───────────────────
    combined_mms = {
        stream: np.lib.format.open_memmap(
            out_paths[stream]["combined"], mode="w+", dtype="float32",
            shape=(total_tokens, d_model))
        for stream in streams
    }

    base_last_lists = {stream: [] for stream in streams}
    tom_last_lists  = {stream: [] for stream in streams}
    row = 0

    for i, (system, user) in enumerate(all_prompts):
        h_base = _extract_all_tokens(model, tokenizer, system, user,
                                     layer_modules, device)
        h_tom  = _extract_all_tokens(model, tokenizer, system,
                                     user + tom_suffix, layer_modules, device)

        seq_base = h_base[streams[0]].shape[0]
        for stream in streams:
            t = h_base[stream]
            combined_mms[stream][row:row + seq_base] = \
                (t.cpu() if t.is_cuda else t).numpy()
        row += seq_base

        seq_tom = h_tom[streams[0]].shape[0]
        for stream in streams:
            t = h_tom[stream]
            combined_mms[stream][row:row + seq_tom] = \
                (t.cpu() if t.is_cuda else t).numpy()
        row += seq_tom

        for stream in streams:
            base_last_lists[stream].append(h_base[stream][-1].cpu())
            tom_last_lists[stream].append(h_tom[stream][-1].cpu())

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_prompts)} prompts processed  "
                  f"({row}/{total_tokens} rows)", flush=True)

    for stream in streams:
        combined_mms[stream].flush()
    del combined_mms

    # ── 5. save paired last-token arrays ────────────────────────────────────
    print(f"\nSaved:", flush=True)
    for stream in streams:
        base_last = torch.stack(base_last_lists[stream]).numpy()
        tom_last  = torch.stack(tom_last_lists[stream]).numpy()
        np.save(out_paths[stream]["base_last"], base_last)
        np.save(out_paths[stream]["tom_last"],  tom_last)

        print(f"  [{stream}] {out_paths[stream]['combined']}   "
              f"[{total_tokens}, {d_model}]  (SAE training: base + ToM tokens)")
        print(f"  [{stream}] {out_paths[stream]['base_last']}  {base_last.shape}  "
              f"(paired base last-token)")
        print(f"  [{stream}] {out_paths[stream]['tom_last']}   {tom_last.shape}  "
              f"(paired ToM last-token)")


if __name__ == "__main__":
    main()
