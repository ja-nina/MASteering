"""
Ablate persona LoRA vs task LoRA using saved checkpoints.

Two modes:

  Sampling mode (default)
    Generates N samples from a fixed prompt under each adapter condition.

  Game mode (--eval-game GAME_ID)
    Runs a full game episode for each adapter condition and reports per-turn
    SVD probe activations for both the trainee and the opponent — the same
    metrics collected during training.

Adapter conditions tested (auto-detected from checkpoint):
  base            — no adapters active
  adapter_a only  — persona LoRA  (SVD-constrained, interpretable axes)
  adapter_b only  — task LoRA     (free weights, unconstrained)
  both            — adapter_a + adapter_b

Checkpoint layout (matches train_lora.py):
  <ckpt>/adapter_a/   PEFT adapter_a weights
  <ckpt>/adapter_b/   PEFT adapter_b weights
  <ckpt>/             tokenizer (falls back to --model id if absent)

Usage:
    # Sampling mode
    python scripts/ablate_loras.py \\
        --checkpoint outputs/lora/nothink/persona_ipd/checkpoint_step00500 \\
        --probe-basis data/svd_basis/qwen3-4b-residual.pt

    # Game mode — full episode with per-turn probe metrics
    python scripts/ablate_loras.py \\
        --checkpoint outputs/lora/nothink/persona_ipd/checkpoint_step00500 \\
        --eval-game IteratedPrisonersDilemma-v0 \\
        --probe-basis data/svd_basis/qwen3-4b-residual.pt \\
        --probe-layer 35

    # Isolate one adapter
    python scripts/ablate_loras.py \\
        --checkpoint outputs/lora/nothink/persona_ipd/final \\
        --adapter-b-only --eval-game IteratedPrisonersDilemma-v0 \\
        --probe-basis data/svd_basis/qwen3-4b-residual.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from testbed.training.generation_utils import (
    STRUCTURED_FORMAT_INSTRUCTION, _extract_action,
)
from testbed.training.rollout import TRAINEE_ID, OPPONENT_ID


# ─── defaults ────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM = (
    "You are a strategic game player. Think carefully before acting — "
    "thorough reasoning leads to better decisions."
)

DEFAULT_USER = (
    "[GAME] Welcome to the Iterated Prisoner's Dilemma.\n"
    "You are Player 1. In each round both players simultaneously choose "
    "to Cooperate or Defect.\n"
    "  Both cooperate → each scores 3 pts\n"
    "  Both defect    → each scores 1 pt\n"
    "  One defects    → defector 5 pts, cooperator 0 pts\n\n"
    "This is Round 1. Please reply with '[Cooperate]' or '[Defect]'."
)

W = 72   # display width


# ─── adapter helpers ──────────────────────────────────────────────────────────

def _set_adapters(model, names: List[str]) -> None:
    for module in model.modules():
        if hasattr(module, "set_adapter"):
            module.set_adapter(names)


def _enter_condition(peft_model, adapter_names):
    """Activate adapter condition. Returns a context object if base (None names)."""
    if peft_model is None:
        return None
    if adapter_names is None:
        ctx = peft_model.disable_adapter()
        ctx.__enter__()
        return ctx
    _set_adapters(peft_model, adapter_names)
    return None


def _exit_condition(ctx):
    if ctx is not None:
        ctx.__exit__(None, None, None)


# ─── generation ───────────────────────────────────────────────────────────────

def _build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system + STRUCTURED_FORMAT_INSTRUCTION},
        {"role": "user",   "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def _generate(model, tokenizer, system: str, user: str,
              max_new_tokens: int, device: str) -> str:
    prompt = _build_prompt(tokenizer, system, user)
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ─── probe ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _probe_text(model, tokenizer, text: str, probe, device: str,
                with_adapters: bool = False) -> Dict:
    """Forward pass over `text` with SVD probe hooks.

    with_adapters=False (default): disable all adapters during the pass so the
      probe reads the base model's representation of the text — comparable
      across conditions.
    with_adapters=True: probe WITH whatever adapters are currently active —
      use this to measure the trainee's internal state under the adapter.
    """
    hooks_spec, get_result = probe.make_hook()
    handles = []
    for path, hook_fn in hooks_spec:
        module = model
        for part in path.split("."):
            module = getattr(module, part)
        handles.append(module.register_forward_hook(hook_fn))

    try:
        ids = tokenizer(text, return_tensors="pt").to(device)
        if not with_adapters and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                model(**ids)
        else:
            model(**ids)
    finally:
        for h in handles:
            h.remove()

    return get_result()


def _fmt_probe(scores: Dict, probe, layer: int, indent: int = 4) -> str:
    pad = " " * indent
    ld = scores.get(str(layer), {})
    z = ld.get("z")
    if not z:
        return f"{pad}[probe] no z vectors at layer {layer}"
    top = probe.rank_traits(z, layer)
    lines = [f"{pad}probe layer {layer}:"]
    for slug, sim in top[:5]:
        bar = "█" * max(1, int(abs(sim) * 24))
        sign = "+" if sim >= 0 else "-"
        lines.append(f"{pad}  {slug:22s} {sign}{abs(sim):.3f}  {bar}")
    return "\n".join(lines)


def _mean_trait_score(scores: Dict, probe, layer: int) -> Optional[float]:
    """Mean cosine-sim of the top trait at the given layer, or None."""
    ld = scores.get(str(layer), {})
    z = ld.get("z")
    if not z:
        return None
    top = probe.rank_traits(z, layer)
    return top[0][1] if top else None


# ─── mode 1: sampling ─────────────────────────────────────────────────────────

def run_sampling(model, tokenizer, probe, peft_model, conditions,
                 system, user, n, max_new_tokens, device, probe_layer):
    print(f"\n{'═' * W}")
    print("ABLATION — SAMPLING MODE")
    print(f"  system : {system[:80]}{'…' if len(system) > 80 else ''}")
    print(f"  user   : {user[:100]}{'…' if len(user) > 100 else ''}")
    print(f"{'═' * W}\n")

    for label, adapter_names in conditions:
        print(f"\n{'╔' + '═' * (W - 2) + '╗'}")
        print(f"║  {label:<{W - 4}}║")
        print(f"{'╚' + '═' * (W - 2) + '╝'}")

        ctx = _enter_condition(peft_model, adapter_names)
        for i in range(n):
            print(f"\n─── sample {i + 1} / {n} ───")
            text = _generate(model, tokenizer, system, user, max_new_tokens, device)
            print(text)
            if probe is not None:
                scores = _probe_text(model, tokenizer, text, probe, device,
                                     with_adapters=(adapter_names is not None))
                print(_fmt_probe(scores, probe, probe_layer))
        _exit_condition(ctx)

    print(f"\n{'═' * W}")
    print("[ablate] done.")


# ─── mode 2: game episode ─────────────────────────────────────────────────────

def _run_episode(
    model, tokenizer, probe,
    peft_model, adapter_names,
    game_id, num_players, max_turns,
    system, max_new_tokens, device, probe_layer,
) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Run one full game episode.

    Trainee  (TRAINEE_ID=1): generates with the specified adapter condition;
                             probed WITH adapters (measures trainee's internal state).
    Opponent (OPPONENT_ID=0): always generates with base model (adapters off);
                             probed without adapters.

    Returns (turn_records, game_rewards).
    Each turn_record: {player_id, is_trainee, obs, action, action_text,
                       probe_scores, probe_layer_z}
    """
    import textarena as ta
    env = ta.make(game_id)
    env.reset(num_players=num_players)

    records: List[Dict] = []
    turn_count = 0

    while turn_count < max_turns:
        player_id, obs_str = env.get_observation()
        is_trainee = (player_id == TRAINEE_ID)

        # Set adapter condition for this player
        if is_trainee:
            ctx = _enter_condition(peft_model, adapter_names)
        else:
            # Opponent always uses base model
            ctx = _enter_condition(peft_model, None)

        action = _generate(model, tokenizer, system, obs_str, max_new_tokens, device)
        action_text = _extract_action(action)

        # Probe: trainee probed WITH adapters, opponent probed without
        probe_scores = None
        if probe is not None:
            probe_scores = _probe_text(
                model, tokenizer, action, probe, device,
                with_adapters=(is_trainee and adapter_names is not None),
            )

        _exit_condition(ctx)

        records.append({
            "turn": turn_count + 1,
            "player_id": player_id,
            "is_trainee": is_trainee,
            "obs": obs_str,
            "action": action,
            "action_text": action_text,
            "probe_scores": probe_scores,
        })

        done, _ = env.step(action_text)
        turn_count += 1
        if done:
            break

    try:
        ta_rewards, _ = env.close()
    except Exception:
        ta_rewards = None
    return records, ta_rewards


def _print_episode(records, ta_rewards, probe, probe_layer, label):
    print(f"\n{'╔' + '═' * (W - 2) + '╗'}")
    print(f"║  {label:<{W - 4}}║")
    print(f"{'╚' + '═' * (W - 2) + '╝'}")

    for rec in records:
        role = "TRAINEE" if rec["is_trainee"] else "OPPONENT"
        role_marker = "▶" if rec["is_trainee"] else "◀"
        print(f"\n  {role_marker} Turn {rec['turn']} | {role} (Player {rec['player_id']})")
        at = rec["action_text"]
        print(f"    action : {at[:120]}{'…' if len(at) > 120 else ''}")

        if rec["probe_scores"] and probe is not None:
            print(_fmt_probe(rec["probe_scores"], probe, probe_layer, indent=4))

    # Summary table: mean top-trait score per player
    if probe is not None:
        trainee_scores = [
            _mean_trait_score(r["probe_scores"], probe, probe_layer)
            for r in records if r["is_trainee"] and r["probe_scores"]
        ]
        opp_scores = [
            _mean_trait_score(r["probe_scores"], probe, probe_layer)
            for r in records if not r["is_trainee"] and r["probe_scores"]
        ]
        trainee_scores = [s for s in trainee_scores if s is not None]
        opp_scores     = [s for s in opp_scores if s is not None]
        print(f"\n  {'─' * (W - 4)}")
        if trainee_scores:
            print(f"  mean top-trait (trainee)  : {sum(trainee_scores)/len(trainee_scores):+.3f}")
        if opp_scores:
            print(f"  mean top-trait (opponent) : {sum(opp_scores)/len(opp_scores):+.3f}")

    if ta_rewards:
        print(f"\n  Game outcome:")
        for pid, score in sorted(ta_rewards.items()):
            role = "trainee" if pid == TRAINEE_ID else "opponent"
            print(f"    Player {pid} ({role}): {score:.1f} pts")


def run_game_eval(model, tokenizer, probe, peft_model, conditions,
                  game_id, num_players, max_turns,
                  system, max_new_tokens, device, probe_layer):
    print(f"\n{'═' * W}")
    print(f"ABLATION — GAME MODE   ({game_id})")
    print(f"  system : {system[:80]}{'…' if len(system) > 80 else ''}")
    print(f"  trainee_id={TRAINEE_ID}  opponent_id={OPPONENT_ID}"
          f"  max_turns={max_turns}")
    print(f"{'═' * W}")

    for label, adapter_names in conditions:
        records, ta_rewards = _run_episode(
            model, tokenizer, probe, peft_model, adapter_names,
            game_id, num_players, max_turns,
            system, max_new_tokens, device, probe_layer,
        )
        _print_episode(records, ta_rewards, probe, probe_layer, label)

    print(f"\n{'═' * W}")
    print("[ablate] done.")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Ablate persona vs task LoRA")
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint dir (contains adapter_a/ and/or adapter_b/)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda")

    # Adapter selection
    ap.add_argument("--adapter-a-only", action="store_true",
                    help="Only load/test adapter_a (persona LoRA)")
    ap.add_argument("--adapter-b-only", action="store_true",
                    help="Only load/test adapter_b (task LoRA)")

    # Probe
    ap.add_argument("--probe-basis", default=None,
                    help="Path to SVD residual basis .pt (e.g. data/svd_basis/qwen3-4b-residual.pt)")
    ap.add_argument("--probe-layer", type=int, default=35)
    ap.add_argument("--probe-hook",  default="residual")

    # Sampling mode
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--user-prompt",   default=None)
    ap.add_argument("--n", type=int, default=2, help="Samples per condition (sampling mode)")
    ap.add_argument("--max-new-tokens", type=int, default=512)

    # Game mode
    ap.add_argument("--eval-game", default=None, metavar="GAME_ID",
                    help="Run a full game episode per condition instead of sampling "
                         "(e.g. IteratedPrisonersDilemma-v0)")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--max-turns",   type=int, default=50)

    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    has_a = (ckpt / "adapter_a").exists() and not args.adapter_b_only
    has_b = (ckpt / "adapter_b").exists() and not args.adapter_a_only

    if not has_a and not has_b:
        sys.exit(
            f"[ablate] Neither adapter_a/ nor adapter_b/ found under {ckpt}.\n"
            f"         Check --checkpoint is a step or final checkpoint dir."
        )

    device = args.device

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_dir = str(ckpt) if (ckpt / "tokenizer_config.json").exists() else args.model
    print(f"[ablate] tokenizer  ← {tok_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    # ── Base model ────────────────────────────────────────────────────────────
    print(f"[ablate] base model ← {args.model}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # ── Adapters ──────────────────────────────────────────────────────────────
    from peft import PeftModel
    peft_model: Optional[PeftModel] = None

    if has_a:
        print(f"[ablate] adapter_a  ← {ckpt / 'adapter_a'}", flush=True)
        peft_model = PeftModel.from_pretrained(
            base_model, str(ckpt / "adapter_a"), adapter_name="adapter_a",
        )
    if has_b:
        print(f"[ablate] adapter_b  ← {ckpt / 'adapter_b'}", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                base_model, str(ckpt / "adapter_b"), adapter_name="adapter_b",
            )
        else:
            peft_model.load_adapter(str(ckpt / "adapter_b"), adapter_name="adapter_b")

    model = (peft_model if peft_model is not None else base_model).to(device)
    model.eval()

    # ── Probe ─────────────────────────────────────────────────────────────────
    probe = None
    if args.probe_basis:
        print(f"[ablate] probe      ← {args.probe_basis}", flush=True)
        from testbed.probing.svd_probe import SVDPersonaProbe
        probe = SVDPersonaProbe(basis_path=args.probe_basis, hook=args.probe_hook)

    # ── Conditions ────────────────────────────────────────────────────────────
    conditions: List[Tuple[str, Optional[List[str]]]] = [
        ("base  (no adapters)", None),
    ]
    if has_a:
        conditions.append(("adapter_a only  [persona LoRA — SVD-constrained]", ["adapter_a"]))
    if has_b:
        conditions.append(("adapter_b only  [task LoRA — free weights]", ["adapter_b"]))
    if has_a and has_b:
        conditions.append(("both  [adapter_a + adapter_b]", ["adapter_a", "adapter_b"]))

    # ── Dispatch ──────────────────────────────────────────────────────────────
    system = args.system_prompt or DEFAULT_SYSTEM

    if args.eval_game:
        run_game_eval(
            model, tokenizer, probe, peft_model, conditions,
            game_id=args.eval_game,
            num_players=args.num_players,
            max_turns=args.max_turns,
            system=system,
            max_new_tokens=args.max_new_tokens,
            device=device,
            probe_layer=args.probe_layer,
        )
    else:
        user = args.user_prompt or DEFAULT_USER
        run_sampling(
            model, tokenizer, probe, peft_model, conditions,
            system=system,
            user=user,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
            device=device,
            probe_layer=args.probe_layer,
        )


if __name__ == "__main__":
    main()
