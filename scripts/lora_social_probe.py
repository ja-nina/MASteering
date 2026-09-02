"""LoRA social probe: what does the trainee activate to steer the opponent?

Setup
-----
  Player 0 (trainee) : no-persona LoRA-adapted Qwen3-4B
  Player 1 (opponent): same base model, adapter disabled (self-play)

Both players are probed on ALL 53 trait directions at every turn.
The resulting per-episode correlation matrix reveals the implicit social
strategy the RL training discovered — without being told what persona to adopt.

Key output
----------
  wandb metrics (per episode):
    trainee/<trait>_cos_sim  — trainee's activation alignment with every trait
    opponent/<trait>_cos_sim — opponent's activation alignment with every trait
    opponent/target_cos_sim  — the reward signal (same key as training)
  wandb summary:
    correlation/<trait>_vs_target — corr(trainee_trait_i, opponent_target) across turns
    social_grammar table          — sorted ranking, most→least correlated

Usage
-----
    python scripts/lora_social_probe.py \\
        --model Qwen/Qwen3-4B \\
        --lora-dir /scratch/.../loras_ipd_nudge_v3_nopersona/ipdNV3np_max_evil \\
        --target-trait evil \\
        --direction max \\
        --std-vectors-dir /scratch/.../qwen3-4b/bf16 \\
        --amoral-vectors-dir /scratch/.../qwen3-4b-amoral-roleplay/bf16 \\
        --n-episodes 10 \\
        --wandb-project lora-social-probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from testbed.training.generation_utils import STRUCTURED_FORMAT_INSTRUCTION, _extract_action

W = 72

DEFAULT_SYSTEM = (
    "You are a strategic game player. Think carefully before acting — "
    "thorough reasoning leads to better decisions."
)

TRAINEE_PLAYER_ID  = 0
OPPONENT_PLAYER_ID = 1

# Traits whose vectors live in the amoral-roleplay extraction
AMORAL_TRAITS = {
    "charismatic", "curious", "directive-stance", "epistemic-humility",
    "evil", "exploratory-stance", "humorous", "optimistic",
    "over-pathologizing", "protocol-rigid-checklist-driven",
    "repair-accountability", "suggestible-leading",
}


# ─── LoRA checkpoint discovery ────────────────────────────────────────────────

def find_latest_checkpoint(lora_dir: Path) -> Path:
    """Return path to the latest checkpoint_stepNNNNN/adapter_b/ inside lora_dir."""
    checkpoints = sorted(lora_dir.glob("checkpoint_step*"), key=lambda p: p.name)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_step* dirs found in {lora_dir}")
    latest = checkpoints[-1]
    adapter_dir = latest / "adapter_b"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter_b/ not found under {latest}")
    print(f"[social_probe] checkpoint : {latest.name}  →  {adapter_dir}")
    return adapter_dir


# ─── probe loading ────────────────────────────────────────────────────────────

def load_all_probes(
    std_dir: Path,
    amoral_dir: Path,
    layer_start: int,
) -> Dict[str, "RawCosineProbe"]:
    """Load RawCosineProbe for every trait found in the vector directories."""
    from testbed.probing.raw_cosine_probe import RawCosineProbe

    probes: Dict[str, "RawCosineProbe"] = {}
    for pt_file in sorted(std_dir.glob("*_raw.pt")):
        trait = pt_file.stem.replace("_raw", "")
        try:
            probes[trait] = RawCosineProbe(str(pt_file), hook="residual", layer_start=layer_start)
        except Exception as e:
            print(f"[social_probe] skip {trait} (std)  : {e}")

    for pt_file in sorted(amoral_dir.glob("*_raw.pt")):
        trait = pt_file.stem.replace("_raw", "")
        if trait not in probes:  # amoral is a fallback for these traits
            try:
                probes[trait] = RawCosineProbe(str(pt_file), hook="residual", layer_start=layer_start)
            except Exception as e:
                print(f"[social_probe] skip {trait} (amoral): {e}")

    print(f"[social_probe] loaded {len(probes)} trait probes")
    return probes


# ─── multi-direction probing ──────────────────────────────────────────────────

@torch.no_grad()
def _probe_all_in_context(
    model,
    tokenizer,
    system: str,
    user: str,
    response: str,
    probes: Dict[str, "RawCosineProbe"],
    layer_start: int,
    device: str,
    collect_raw: bool = False,
) -> Tuple[Dict[str, float], Dict[int, torch.Tensor]]:
    """One forward pass; compute mean cos sim (layers >= layer_start) for all traits.

    Returns:
        cos_sims : {trait_name: mean_cos_sim_float}
        raw_acts : {layer_int: mean_hidden_state float16 cpu tensor [hidden_dim]}
                   only populated when collect_raw=True, else empty dict
    """
    # ── token boundary ────────────────────────────────────────────────────────
    context_msgs = [
        {"role": "system", "content": system + STRUCTURED_FORMAT_INSTRUCTION},
        {"role": "user",   "content": user},
    ]
    try:
        ctx_text = tokenizer.apply_chat_template(
            context_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        ctx_text = tokenizer.apply_chat_template(
            context_msgs, tokenize=False, add_generation_prompt=True,
        )
    context_len = tokenizer(ctx_text, return_tensors="pt")["input_ids"].shape[1]

    full_msgs = [
        {"role": "system",    "content": system + STRUCTURED_FORMAT_INSTRUCTION},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": response},
    ]
    try:
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
    except TypeError:
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False,
        )
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    # ── build layer → {trait → direction_tensor} lookup ──────────────────────
    # Only include layers that are >= layer_start for at least one probe
    layer_to_dirs: Dict[int, Dict[str, torch.Tensor]] = {}
    for trait, probe in probes.items():
        for l, d in probe.directions.items():
            if l >= layer_start:
                if l not in layer_to_dirs:
                    layer_to_dirs[l] = {}
                layer_to_dirs[l][trait] = d

    # ── accumulate: {trait -> [cos_sim per active layer]} ────────────────────
    trait_layer_sims: Dict[str, List[float]] = {t: [] for t in probes}
    raw_acts: Dict[int, torch.Tensor] = {}   # layer -> {"mean": ..., "tokens": ...}
    # response token ids — only populated when collect_raw is set
    response_token_ids: Optional[torch.Tensor] = None
    handles = []

    bm = getattr(model, "base_model", None)
    hook_root = bm.model if (bm is not None and hasattr(bm, "model")) else model

    def _make_hook(layer_int: int, dirs: Dict[str, torch.Tensor]):
        def _h(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            resp_h = h[0, context_len:, :].detach().float()
            if resp_h.shape[0] == 0:
                return
            mean_h = resp_h.mean(dim=0)
            if collect_raw:
                # "mean" key: averaged vector [hidden_dim] — compact, used for cos sim
                # "tokens" key: per-token matrix [n_tokens, hidden_dim] — full resolution
                raw_acts[layer_int] = {
                    "mean":   mean_h.cpu().half(),
                    "tokens": resp_h.cpu().half() if collect_raw == "full" else None,
                }
            h_norm = mean_h.norm().clamp(min=1e-8)
            for trait, d in dirs.items():
                cos = (d.to(mean_h.device) @ mean_h / h_norm).item()
                trait_layer_sims[trait].append(cos)
        return _h

    for l, dirs in layer_to_dirs.items():
        path = f"model.layers.{l}"
        mod = hook_root
        for part in path.split("."):
            mod = getattr(mod, part)
        handles.append(mod.register_forward_hook(_make_hook(l, dirs)))

    if collect_raw:
        # shape [n_response_tokens] — lets you decode each position later
        response_token_ids = inputs["input_ids"][0, context_len:].cpu()

    try:
        model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # Return per-layer breakdown: {trait: {layer_int: cos_sim}}
    # trait_layer_sims[trait] accumulated in hook-firing order = sorted(layer_to_dirs)
    active_layers = sorted(layer_to_dirs.keys())
    cos_sims: Dict[str, Dict[int, float]] = {}
    for trait, sims in trait_layer_sims.items():
        cos_sims[trait] = {
            l: sims[i] for i, l in enumerate(active_layers) if i < len(sims)
        }
    if collect_raw:
        raw_acts["_token_ids"] = response_token_ids  # aligned with "tokens" dim 0
    return cos_sims, raw_acts


# ─── generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate(model, tokenizer, system: str, user: str,
              max_new_tokens: int, device: str) -> str:
    messages = [
        {"role": "system", "content": system + STRUCTURED_FORMAT_INSTRUCTION},
        {"role": "user",   "content": user},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **ids, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
    )
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ─── episode runner ──────────────────────────────────────────────────────────

def run_episode(
    model,
    tokenizer,
    probes: Dict[str, "RawCosineProbe"],
    target_trait: str,
    layer_start: int,
    game_id: str,
    num_players: int,
    max_turns: int,
    system: str,
    max_new_tokens: int,
    device: str,
    seed: Optional[int] = None,
    collect_raw: bool = False,
) -> List[Dict]:
    import textarena as ta
    env = ta.make(game_id)
    try:
        env.reset(num_players=num_players, seed=seed)
    except TypeError:
        env.reset(num_players=num_players)

    records: List[Dict] = []
    turn_count = 0

    while turn_count < max_turns:
        player_id, obs_str = env.get_observation()
        is_trainee = (player_id == TRAINEE_PLAYER_ID)
        role = "trainee" if is_trainee else "opponent"
        print(f"  [turn {turn_count+1:2d}] {role} ...", end=" ", flush=True)

        t0 = time.time()

        if is_trainee:
            # LoRA enabled for trainee
            action = _generate(model, tokenizer, system, obs_str, max_new_tokens, device)
            all_cos, raw = _probe_all_in_context(
                model, tokenizer, system, obs_str, action, probes, layer_start, device,
                collect_raw=collect_raw,
            )
            trainee_cos, trainee_raw = all_cos, raw
            opponent_cos, opponent_raw = {}, {}
        else:
            # Disable LoRA for opponent — same weights, no adapter
            with model.disable_adapter():
                action = _generate(model, tokenizer, system, obs_str, max_new_tokens, device)
                all_cos, raw = _probe_all_in_context(
                    model, tokenizer, system, obs_str, action, probes, layer_start, device,
                    collect_raw=collect_raw,
                )
            trainee_cos, trainee_raw = {}, {}
            opponent_cos, opponent_raw = all_cos, raw

        elapsed = time.time() - t0
        # target_cos: mean across layers for display only
        target_layer_dict = (trainee_cos if is_trainee else opponent_cos).get(target_trait, {})
        target_cos = (sum(target_layer_dict.values()) / len(target_layer_dict)
                      if target_layer_dict else float("nan"))
        print(f"{elapsed:.2f}s  {target_trait}={target_cos:+.3f}", flush=True)

        action_text = _extract_action(action)
        records.append({
            "turn":         turn_count + 1,
            "player_id":    player_id,
            "is_trainee":   is_trainee,
            "action":       action,
            "action_text":  action_text,
            "trainee_cos":  trainee_cos,   # {trait: {layer: float}} or {} if opponent's turn
            "opponent_cos": opponent_cos,  # {trait: {layer: float}} or {} if trainee's turn
            "trainee_raw":  trainee_raw,   # {layer_int: {...}} or {}
            "opponent_raw": opponent_raw,  # {layer_int: {...}} or {}
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


# ─── main ────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Probe LoRA trainee + opponent with all 53 trait directions"
    )
    p.add_argument("--model",             required=True,  help="Base model HF id")
    p.add_argument("--lora-dir",          required=True,
                   help="Root LoRA dir (e.g. .../ipdNV3np_max_evil); "
                        "script picks the latest checkpoint_step* automatically")
    p.add_argument("--target-trait",      required=True)
    p.add_argument("--direction",         default="max", choices=["max", "min"])
    p.add_argument("--std-vectors-dir",   required=True,
                   help="Dir with standard *_raw.pt files")
    p.add_argument("--amoral-vectors-dir", required=True,
                   help="Dir with amoral-roleplay *_raw.pt files")
    p.add_argument("--layer-start",       default=20, type=int)
    p.add_argument("--n-episodes",        default=10, type=int)
    p.add_argument("--game",              default="IteratedPrisonersDilemma-v0")
    p.add_argument("--num-players",       default=2, type=int)
    p.add_argument("--max-turns",         default=50, type=int)
    p.add_argument("--max-new-tokens",    default=300, type=int)
    p.add_argument("--device",            default="cuda")
    p.add_argument("--system",            default=DEFAULT_SYSTEM)
    p.add_argument("--output",            default=None)
    p.add_argument("--wandb-project",     default="lora-social-probe")
    p.add_argument("--wandb-run-name",    default=None)
    p.add_argument("--activations-dir",        default=None,
                   help="If set, save raw hidden states (float16) to "
                        "<dir>/sp_<trait>_<direction>.pt for offline analysis")
    p.add_argument("--full-token-episodes",    default=1, type=int,
                   help="Save per-token activations (large) for this many episodes; "
                        "remaining episodes get mean-only (default: 1)")
    args = p.parse_args(argv)

    lora_dir   = Path(args.lora_dir)
    std_dir    = Path(args.std_vectors_dir)
    amoral_dir = Path(args.amoral_vectors_dir)
    adapter_dir = find_latest_checkpoint(lora_dir)

    # ── load base model + LoRA ────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[social_probe] base model : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    print(f"[social_probe] LoRA loaded from {adapter_dir}")

    # ── load all trait probes ─────────────────────────────────────────────────
    probes = load_all_probes(std_dir, amoral_dir, args.layer_start)
    if args.target_trait not in probes:
        print(f"[social_probe] WARNING: target trait '{args.target_trait}' not in probes")

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project and args.wandb_project != "disabled":
        try:
            import wandb
            run_name = args.wandb_run_name or f"sp_{args.target_trait}_{args.direction}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "target_trait":  args.target_trait,
                    "direction":     args.direction,
                    "lora_dir":      str(lora_dir),
                    "adapter_dir":   str(adapter_dir),
                    "base_model":    args.model,
                    "n_episodes":    args.n_episodes,
                    "n_traits":      len(probes),
                    "layer_start":   args.layer_start,
                    "game":          args.game,
                },
            )
            print(f"[social_probe] wandb run: {wandb_run.url}")
        except Exception as e:
            print(f"[social_probe] wandb init failed: {e}")

    # ── run episodes ──────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f" LoRA Social Probe: {args.target_trait} ({args.direction}), {args.n_episodes} episodes")
    print(f"{'═'*W}\n")

    all_records: List[Dict] = []
    episode_results = []

    out_fh = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(out_path, "w")

    collect_raw = args.activations_dir is not None
    n_full_eps  = args.full_token_episodes if collect_raw else 0
    if collect_raw:
        acts_dir = Path(args.activations_dir)
        acts_dir.mkdir(parents=True, exist_ok=True)
        # Accumulate raw tensors: list of {layer: tensor} per turn, per player
        all_trainee_raws: List[Dict[int, torch.Tensor]] = []
        all_opponent_raws: List[Dict[int, torch.Tensor]] = []
        # Metadata aligned with the raw lists
        trainee_meta: List[Dict] = []   # {episode, turn}
        opponent_meta: List[Dict] = []

    # accumulators for correlation per (trait, layer) — keyed by episode mean
    # trainee_layer_accum[trait][layer] = [ep_mean, ep_mean, ...]
    trainee_layer_accum: Dict[str, Dict[int, List[float]]] = {t: {} for t in probes}
    # opponent_target_layer_accum[layer] = [ep_mean, ep_mean, ...]
    opponent_target_layer_accum: Dict[int, List[float]] = {}

    for ep_idx in range(args.n_episodes):
        print(f"── Episode {ep_idx + 1} / {args.n_episodes} ──")
        t0 = time.time()

        records, ta_rewards = run_episode(
            model=model, tokenizer=tokenizer,
            probes=probes, target_trait=args.target_trait,
            layer_start=args.layer_start,
            game_id=args.game, num_players=args.num_players,
            max_turns=args.max_turns, system=args.system,
            max_new_tokens=args.max_new_tokens, device=args.device,
            seed=ep_idx,
            collect_raw="full" if (collect_raw and ep_idx < n_full_eps) else collect_raw,
        )
        if collect_raw:
            for r in records:
                if r["is_trainee"] and r["trainee_raw"]:
                    all_trainee_raws.append(r["trainee_raw"])
                    trainee_meta.append({"episode": ep_idx + 1, "turn": r["turn"]})
                elif not r["is_trainee"] and r["opponent_raw"]:
                    all_opponent_raws.append(r["opponent_raw"])
                    opponent_meta.append({"episode": ep_idx + 1, "turn": r["turn"]})
        elapsed = time.time() - t0
        all_records.extend(records)

        # ── episode averages per (trait, layer) ──────────────────────────────
        # ep_trainee_layer[trait][layer] = mean cos sim across trainee turns
        ep_trainee_layer: Dict[str, Dict[int, float]] = {}
        ep_opponent_layer: Dict[str, Dict[int, float]] = {}

        for trait in probes:
            # collect per-layer lists across turns
            t_by_layer: Dict[int, List[float]] = {}
            for r in records:
                if r["is_trainee"] and trait in r["trainee_cos"]:
                    for layer, v in r["trainee_cos"][trait].items():
                        if v == v:  # not nan
                            t_by_layer.setdefault(layer, []).append(v)
            if t_by_layer:
                ep_trainee_layer[trait] = {l: sum(vs)/len(vs) for l, vs in t_by_layer.items()}

            o_by_layer: Dict[int, List[float]] = {}
            for r in records:
                if not r["is_trainee"] and trait in r["opponent_cos"]:
                    for layer, v in r["opponent_cos"][trait].items():
                        if v == v:
                            o_by_layer.setdefault(layer, []).append(v)
            if o_by_layer:
                ep_opponent_layer[trait] = {l: sum(vs)/len(vs) for l, vs in o_by_layer.items()}

        # scalar target cos (mean across layers) for display and legacy wandb key
        target_layer_ep = ep_opponent_layer.get(args.target_trait, {})
        ep_target_cos = (sum(target_layer_ep.values()) / len(target_layer_ep)
                         if target_layer_ep else None)

        # accumulate per (trait, layer) for correlation
        for trait, layer_means in ep_trainee_layer.items():
            for layer, v in layer_means.items():
                trainee_layer_accum[trait].setdefault(layer, []).append(v)
        for layer, v in target_layer_ep.items():
            opponent_target_layer_accum.setdefault(layer, []).append(v)

        tgt_str = f"{ep_target_cos:+.3f}" if ep_target_cos is not None else "n/a"
        print(f"  turns={len(records)}  opponent_target_cos={tgt_str}  ({elapsed:.1f}s)")

        if wandb_run is not None:
            log = {"episode": ep_idx + 1, "elapsed_s": elapsed, "n_turns": len(records)}
            if ep_target_cos is not None:
                log["opponent/target_cos_sim"] = ep_target_cos
                log["reward/raw_cos_sim_mean"] = ep_target_cos
            # log per-layer means for trainee and opponent (all traits)
            for trait, layer_means in ep_trainee_layer.items():
                for layer, v in layer_means.items():
                    log[f"trainee/{trait}_L{layer:02d}"] = v
            for trait, layer_means in ep_opponent_layer.items():
                for layer, v in layer_means.items():
                    log[f"opponent/{trait}_L{layer:02d}"] = v
            wandb_run.log(log, step=ep_idx + 1)

        ep_record = {
            "episode":           ep_idx + 1,
            "target_trait":      args.target_trait,
            "direction":         args.direction,
            "n_turns":           len(records),
            "ep_trainee_layer":  ep_trainee_layer,   # {trait: {layer: float}}
            "ep_opponent_layer": ep_opponent_layer,  # {trait: {layer: float}}
            "game_rewards":      {str(k): float(v) for k, v in (ta_rewards or {}).items()},
        }
        episode_results.append(ep_record)
        if out_fh:
            out_fh.write(json.dumps(ep_record) + "\n")
            out_fh.flush()

    # ── save activations ──────────────────────────────────────────────────────
    if collect_raw and all_trainee_raws:
        acts_path = acts_dir / f"sp_{args.target_trait}_{args.direction}.pt"
        torch.save({
            "meta": {
                "target_trait":       args.target_trait,
                "direction":          args.direction,
                "base_model":         args.model,
                "lora_dir":           str(lora_dir),
                "layer_start":        args.layer_start,
                "n_full_token_eps":   n_full_eps,
                "n_episodes":         args.n_episodes,
            },
            # trainee: list[dict[layer_int -> {"mean": [D], "tokens": [T,D] or None}]]
            # one entry per trainee turn across all episodes
            "trainee":  all_trainee_raws,
            "trainee_meta": trainee_meta,
            # opponent: same structure for opponent turns
            "opponent": all_opponent_raws,
            "opponent_meta": opponent_meta,
        }, acts_path)
        size_mb = acts_path.stat().st_size / 1e6
        print(f"[social_probe] activations saved → {acts_path}  ({size_mb:.1f} MB)")

    # ── correlation analysis: per (trait, layer) ─────────────────────────────
    print(f"\n{'═'*W}")
    print(" SOCIAL GRAMMAR — corr(trainee_trait @ layer L, opponent_target @ layer L)")
    print(f"{'═'*W}")

    def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
        n = min(len(xs), len(ys))
        if n < 3:
            return None
        xs, ys = xs[:n], ys[:n]
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx  = sum((x-mx)**2 for x in xs) ** 0.5
        dy  = sum((y-my)**2 for y in ys) ** 0.5
        if dx < 1e-8 or dy < 1e-8:
            return None
        return num / (dx * dy)

    all_layers = sorted(opponent_target_layer_accum.keys())

    # grammar[trait][layer] = pearson correlation (or None)
    grammar: Dict[str, Dict[int, Optional[float]]] = {}
    for trait in probes:
        grammar[trait] = {}
        for layer in all_layers:
            tr_vals  = trainee_layer_accum[trait].get(layer, [])
            opp_vals = opponent_target_layer_accum.get(layer, [])
            grammar[trait][layer] = _pearson(tr_vals, opp_vals)

    # max-abs correlation across layers per trait → for ranking
    trait_max_corr = {
        trait: max((abs(v) for v in layer_dict.values() if v is not None), default=0.0)
        for trait, layer_dict in grammar.items()
    }
    ranked_traits = sorted(probes.keys(), key=lambda t: -trait_max_corr[t])

    # Print top-10 traits with their best layer
    for trait in ranked_traits[:10]:
        best_layer, best_corr = max(
            ((l, v) for l, v in grammar[trait].items() if v is not None),
            key=lambda x: abs(x[1]), default=(None, None)
        )
        if best_corr is None:
            continue
        bar  = "█" * int(abs(best_corr) * 20)
        sign = "+" if best_corr >= 0 else "-"
        print(f"  {sign}{bar:<20s}  {best_corr:+.3f}  {trait}  (best @ layer {best_layer})")

    if wandb_run is not None:
        import wandb as _wandb
        summary = {}

        all_opp_vals = [v for vs in opponent_target_layer_accum.values() for v in vs]
        if all_opp_vals:
            m = sum(all_opp_vals) / len(all_opp_vals)
            summary["summary/mean_target_cos_sim"] = m
            summary["reward/raw_cos_sim_mean"]      = m

        # flat correlation metrics
        for trait, layer_dict in grammar.items():
            for layer, corr in layer_dict.items():
                if corr is not None:
                    summary[f"correlation/{trait}_L{layer:02d}_vs_target"] = corr

        # heatmap table: rows = traits, cols = layers, values = correlation
        if all_layers:
            col_names = ["trainee_trait", "max_abs_corr"] + [f"L{l:02d}" for l in all_layers]
            tbl = _wandb.Table(columns=col_names)
            for trait in ranked_traits:
                row = [trait, round(trait_max_corr[trait], 4)]
                for layer in all_layers:
                    v = grammar[trait].get(layer)
                    row.append(round(v, 4) if v is not None else None)
                tbl.add_data(*row)
            summary["social_grammar_heatmap"] = tbl

        wandb_run.log(summary)
        wandb_run.finish()

    if out_fh:
        out_fh.close()
        print(f"[social_probe] saved → {out_path}")


if __name__ == "__main__":
    main()
