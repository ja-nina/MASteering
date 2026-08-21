"""GRPO (Group Relative Policy Optimisation) for persona LoRA training.

At each trainee turn, K candidate responses were sampled from the same
observation.  The opponent was probed on each → K rewards.  These K rewards
are normalised within the turn group to form per-turn advantages:

    a_k  =  (r_k − μ_turn) / (σ_turn + ε)

Gradients accumulate turn-by-turn (one backward per TurnGroup) so only one
turn's K computation graphs live in GPU memory at once.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch

from testbed.training.rollout import Episode, TurnGroup


def lora_inject_breakdown(
    episode: Episode,
    model,
    probe,
) -> Dict[str, float]:
    """Project adapter_a's lora_A outputs onto trait directions.

    The adapter's contribution to the residual stream is:
        Δh = lora_B @ z    where z = lora_A @ x,   lora_B = Vk[:r].T (frozen)

    So z ∈ ℝʳ encodes the activation in SVD space.  Projecting onto
    C[:, :r] (trait directions truncated to the first r SVD components)
    gives the per-trait cosine-similarity of what adapter_a is injecting.

    Runs one no-grad teacher-forcing pass on candidate-0 of the last turn.
    Returns {trait_slug: mean_cosine_sim_over_layers}.
    """
    if not episode.turn_groups:
        return {}
    record = episode.turn_groups[-1].records[0]
    if record.full_ids is None:
        return {}

    device = next(model.parameters()).device
    captured: Dict[int, torch.Tensor] = {}   # layer_idx → mean z [r]
    handles = []

    def _make_hook(layer_idx: int):
        def hook(module, inp, out):
            # out: [batch=1, seq_len, r]
            captured[layer_idx] = out.detach().float().mean(dim=[0, 1]).cpu()
        return hook

    for name, module in model.named_modules():
        if "o_proj" in name and "lora_A" in name and "adapter_a" in name:
            layer_idx = next(
                (int(p) for p in name.split(".") if p.isdigit()), None
            )
            if layer_idx is not None:
                handles.append(module.register_forward_hook(_make_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(record.full_ids.to(device))
    finally:
        for h in handles:
            h.remove()

    if not captured:
        return {}

    r = next(iter(captured.values())).shape[0]
    slug_cos_sums: Dict[str, float] = {s: 0.0 for s in probe._slugs}
    count = 0
    for layer_idx, z in captured.items():
        if layer_idx not in probe._C:
            continue
        C_r = probe._C[layer_idx].float()[:, :r]   # [N_traits, min(r, n_svd)]
        r_eff = C_r.shape[1]   # may be < r if SVD basis has fewer components than lora rank
        z_eff = z[:r_eff]
        z_norm = z_eff.norm().clamp(min=1e-8)
        C_r_norms = C_r.norm(dim=1).clamp(min=1e-8)
        cos_sims = ((C_r @ z_eff) / (C_r_norms * z_norm)).tolist()
        for slug, sim in zip(probe._slugs, cos_sims):
            slug_cos_sums[slug] += sim
        count += 1

    if count == 0:
        return {}
    return {s: v / count for s, v in slug_cos_sums.items()}


def grpo_step(
    episode: Episode,
    optimizers,
    max_grad_norm: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """One GRPO gradient step over a single episode with K candidates per turn.

    For each TurnGroup:
      - normalise K rewards → K advantages
      - accumulate loss: Σ_k  −log_prob_k × advantage_k
      - backward() immediately → frees this turn's K graphs

    Returns total scalar loss (summed across all turn groups).
    """
    if not episode.turn_groups:
        return 0.0

    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    for opt in optimizers:
        opt.zero_grad()

    total_loss = 0.0

    for tg in episode.turn_groups:
        K = len(tg.rewards)
        if K < 2:
            tg.advantages = [0.0] * K
            continue

        mu  = sum(tg.rewards) / K
        std = (sum((r - mu) ** 2 for r in tg.rewards) / K) ** 0.5
        advantages = [(r - mu) / (std + eps) for r in tg.rewards]
        tg.advantages = advantages

        tg_loss = torch.tensor(0.0)
        for record, adv in zip(tg.records, advantages):
            if record.log_prob is not None:
                tg_loss = tg_loss + (-record.log_prob * float(adv))

        if tg_loss.requires_grad:
            tg_loss.backward()   # frees this turn group's graphs
            total_loss += tg_loss.item()

    all_params = [p for opt in optimizers for pg in opt.param_groups for p in pg["params"]]
    if any(p.grad is not None for p in all_params):
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        for opt in optimizers:
            opt.step()

    return total_loss


def episode_stats(episode: Episode) -> dict:
    """Summary statistics over all turn-group rewards in one episode."""
    all_rewards = [r for tg in episode.turn_groups for r in tg.rewards]
    if not all_rewards:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mu = sum(all_rewards) / len(all_rewards)
    sigma = (sum((r - mu) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
    return {"mean": mu, "std": sigma, "min": min(all_rewards), "max": max(all_rewards)}


def wandb_log_step(
    wandb_run,
    step: int,
    episode: Episode,
    loss: float,
    model,
    probe,
    probe_layer: int,
    use_persona_lora: bool,
    log_transcript_every: int = 50,
    rolling_mean: Optional[float] = None,
    game_outcome: Optional[dict] = None,
    step_time: Optional[float] = None,
    target_trait_slugs: Optional[List[str]] = None,
) -> None:
    """Log rich training diagnostics to wandb."""
    if wandb_run is None:
        return

    import torch

    stats = episode_stats(episode)

    # ── 1. Core scalars ──────────────────────────────────────────────────────
    n_turns = len(episode.turn_groups)
    log = {
        "loss":               loss,
        "reward/mean":        stats["mean"],
        "reward/std":         stats["std"],
        "reward/min":         stats["min"],
        "reward/max":         stats["max"],
        "episode/num_turns":  n_turns,
        "episode/mean_length_chars": (
            sum(len(r.action) for tg in episode.turn_groups for r in tg.records)
            / max(sum(len(tg.records) for tg in episode.turn_groups), 1)
        ),
    }
    if rolling_mean is not None:
        log["reward/rolling_mean_10"] = rolling_mean
    if step_time is not None:
        log["train/step_time_s"] = step_time
    if game_outcome is not None:
        log["episode/trainee_game_reward"] = game_outcome.get("trainee", float("nan"))
        log["episode/opponent_game_reward"] = game_outcome.get("opponent", float("nan"))

    # ── 2. Persona projections at display_layer only (trainee, K=0) ─────────
    # Averaging over all trainee turns keeps this representative without logging
    # all 26 layers × 55 traits = 1430 metrics per step.
    layer_z_sums: dict = {}
    layer_z_counts: dict = {}
    for tg in episode.turn_groups:
        record = tg.records[0]   # sample 0 carries probe_z_all
        if not record.probe_z_all:
            continue
        for layer_key, z_list in record.probe_z_all.items():
            if int(layer_key) != probe_layer:
                continue   # only display_layer
            z = torch.tensor(z_list, dtype=torch.float32)
            if layer_key not in layer_z_sums:
                layer_z_sums[layer_key] = z.clone()
                layer_z_counts[layer_key] = 1
            else:
                layer_z_sums[layer_key] += z
                layer_z_counts[layer_key] += 1

    for layer_key, z_sum in layer_z_sums.items():
        mean_z = z_sum / layer_z_counts[layer_key]
        layer_int = int(layer_key)
        if layer_int in probe._C:
            C = probe._C[layer_int].float()
            z_norm = mean_z.norm().clamp(min=1e-8)
            C_norms = C.norm(dim=1).clamp(min=1e-8)
            sims = (C @ mean_z) / (C_norms * z_norm)
            for slug, sim in zip(probe._slugs, sims.tolist()):
                log[f"persona/{slug}"] = sim
                # Dedicated target-trait panel for quick dashboard glance
                if target_trait_slugs and slug in target_trait_slugs:
                    log[f"target_trait/{slug}"] = sim

    # ── 3. LoRA B-matrix norms ───────────────────────────────────────────────
    for name, param in model.named_parameters():
        if "lora_B" not in name or not param.requires_grad:
            continue
        adapter = ("adapter_a" if "adapter_a" in name
                   else "adapter_b" if "adapter_b" in name else None)
        if adapter is None:
            continue
        parts = name.split(".")
        try:
            layer_idx = next(int(p) for p in parts if p.isdigit())
        except StopIteration:
            continue
        mod = ("o_proj" if "o_proj" in name
               else "q_proj" if "q_proj" in name else "other")
        log[f"lora_norm/{adapter}/layer_{layer_idx}/{mod}"] = param.data.norm().item()

    # ── 4. Gradient norms ────────────────────────────────────────────────────
    for adapter_name in (["adapter_a", "adapter_b"] if use_persona_lora else ["adapter_b"]):
        grad_norms = [
            p.grad.norm().item()
            for n, p in model.named_parameters()
            if adapter_name in n and p.grad is not None
        ]
        if grad_norms:
            log[f"grad_norm/{adapter_name}"] = (
                sum(g ** 2 for g in grad_norms) ** 0.5
            )

    # ── 5. adapter_a persona injection breakdown ─────────────────────────────
    # What traits is adapter_a actually injecting into the residual stream?
    # z = lora_A @ x lives in the r=20 SVD subspace; projecting onto C[:,:r]
    # gives the per-trait cosine-sim of what's being added, averaged over layers.
    if use_persona_lora:
        inject = lora_inject_breakdown(episode, model, probe)
        for slug, sim in inject.items():
            log[f"adapter_a/inject/{slug}"] = sim
            if target_trait_slugs and slug in target_trait_slugs:
                log[f"adapter_a/inject_target/{slug}"] = sim

    # ── 6. Opponent reaction timeseries — K rewards per turn ─────────────────
    # Shows how the K candidate responses at each turn compared against each
    # other in terms of opponent persona shift.
    try:
        import wandb as _wandb
        rows = []
        for turn_idx, tg in enumerate(episode.turn_groups):
            if probe_layer not in probe._C:
                continue
            C = probe._C[probe_layer].float()
            for k, (record, reward, adv) in enumerate(
                zip(tg.records, tg.rewards,
                    tg.advantages or [float("nan")] * len(tg.records))
            ):
                if record.probe_z_opponent:
                    z = torch.tensor(record.probe_z_opponent, dtype=torch.float32)
                    z_norm = z.norm().clamp(min=1e-8)
                    C_norms = C.norm(dim=1).clamp(min=1e-8)
                    sims = (C @ z) / (C_norms * z_norm)
                    for slug, sim in zip(probe._slugs, sims.tolist()):
                        rows.append([turn_idx, k, slug,
                                     round(sim, 4), round(reward, 4),
                                     round(adv, 4) if adv == adv else float("nan")])
        if rows:
            log["opponent_persona/timeseries"] = _wandb.Table(
                columns=["turn", "candidate", "trait",
                         "cosine_sim_opponent", "reward", "advantage"],
                data=rows,
            )
    except Exception:
        pass

    # ── 6. Trainee persona timeseries (random turn's sample 0) ──────────────
    try:
        import wandb as _wandb
        if episode.turn_groups:
            sample_tg = random.choice(episode.turn_groups)
            record = sample_tg.records[0]
            rows = []
            if record.probe_z_all:
                turn_idx = episode.turn_groups.index(sample_tg)
                for layer_key, z_list in record.probe_z_all.items():
                    layer_int = int(layer_key)
                    if layer_int not in probe._C:
                        continue
                    z = torch.tensor(z_list, dtype=torch.float32)
                    C = probe._C[layer_int].float()
                    z_norm = z.norm().clamp(min=1e-8)
                    C_norms = C.norm(dim=1).clamp(min=1e-8)
                    sims = (C @ z) / (C_norms * z_norm)
                    for slug, sim in zip(probe._slugs, sims.tolist()):
                        rows.append([step, turn_idx, int(layer_key), slug, sim])
            if rows:
                log["persona_timeseries/turn_sample"] = _wandb.Table(
                    columns=["step", "turn", "layer", "trait", "cosine_sim"],
                    data=rows,
                )
    except Exception:
        pass

    # ── 7. Transcript HTML — every step (readable inline in wandb) ───────────
    if episode.turn_groups:
        try:
            import wandb as _wandb
            html_parts = [
                "<style>"
                "body{font-family:monospace;font-size:13px;padding:8px}"
                ".turn{border:1px solid #444;border-radius:4px;margin:6px 0;padding:8px}"
                ".turn-header{font-weight:bold;margin-bottom:4px;color:#aaa;font-size:11px}"
                ".obs{color:#888;white-space:pre-wrap;margin-bottom:6px;font-size:11px}"
                ".action{white-space:pre-wrap;color:#eee}"
                ".meta{margin-top:4px;font-size:11px;color:#6cf}"
                ".reward-pos{color:#4f4}</style>"
                "<h3 style='margin:0 0 8px;color:#ccc'>Rollout transcript — "
                f"step {step} &nbsp; ({len(episode.turn_groups)} trainee turns)</h3>"
            ]
            for t_idx, tg in enumerate(episode.turn_groups):
                record = tg.records[0]  # candidate played
                reward = tg.rewards[0]
                adv_str = (f"adv={tg.advantages[0]:+.3f}"
                           if tg.advantages else "")
                best_r = max(tg.rewards)
                top_trait_str = ""
                if record.probe_z_opponent:
                    try:
                        top = probe.rank_traits(record.probe_z_opponent, probe_layer)
                        if top:
                            top_trait_str = f"  top_trait={top[0][0]}:{top[0][1]:+.2f}"
                    except Exception:
                        pass
                r_cls = "reward-pos" if reward > 0 else ""
                obs_snip = (record.obs[-400:].replace("<", "&lt;")
                            .replace(">", "&gt;").replace("\n", "↵\n"))
                action_txt = record.action.replace("<", "&lt;").replace(">", "&gt;")
                html_parts.append(
                    f"<div class='turn'>"
                    f"<div class='turn-header'>TURN {t_idx+1}</div>"
                    f"<div class='obs'>…{obs_snip}</div>"
                    f"<div class='action'>{action_txt}</div>"
                    f"<div class='meta'>"
                    f"<span class='{r_cls}'>r={reward:+.4f}</span> "
                    f"best={best_r:+.4f}  {adv_str}{top_trait_str}"
                    f"</div></div>"
                )
            log["rollout/transcript"] = _wandb.Html("".join(html_parts))
        except Exception:
            pass

    # ── 8. Detailed Table — every N steps (for offline filtering) ────────────
    if step % log_transcript_every == 0 and episode.turn_groups:
        try:
            import wandb as _wandb
            rows = []
            for turn_idx, tg in enumerate(episode.turn_groups):
                for k, (record, reward) in enumerate(zip(tg.records, tg.rewards)):
                    adv = (tg.advantages[k] if tg.advantages else float("nan"))
                    top_trait, top_sim = "", float("nan")
                    if record.probe_z_opponent:
                        try:
                            top = probe.rank_traits(record.probe_z_opponent, probe_layer)
                            if top:
                                top_trait, top_sim = top[0][0], float(top[0][1])
                        except Exception:
                            pass
                    rows.append([
                        turn_idx, k,
                        record.obs[-300:],
                        record.action[:500],
                        round(reward, 4),
                        round(adv, 4) if adv == adv else float("nan"),
                        top_trait,
                        round(top_sim, 4) if top_sim == top_sim else float("nan"),
                    ])
            if rows:
                log["rollout/transcript_table"] = _wandb.Table(
                    columns=["turn", "candidate", "observation_tail",
                             "action", "reward", "advantage",
                             "top_trait", "top_sim"],
                    data=rows,
                )
        except Exception:
            pass

    wandb_run.log(log, step=step)
