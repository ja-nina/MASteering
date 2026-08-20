"""GRPO (Group Relative Policy Optimization) for dual persona LoRA training.

Collects K rollouts per update step (a "group"), normalises rewards within
the group to form advantages, and does one gradient step per adapter.

Reference: DeepSeek-R1 (Shao et al. 2024) — same algorithm, adapted for
multi-turn self-play with a dense SVD cosine-similarity reward.

Advantage formula:
    a_i = (mean_reward_i - μ_group) / (σ_group + ε)

where mean_reward_i is the mean per-turn reward across episode i.
All turns within episode i share the same advantage a_i.
"""
from __future__ import annotations

from typing import List, Optional

import torch


def grpo_step(
    episode_records: List[List],          # [K][T] — K episodes, each a list of TurnRecords
    episode_rewards: List[List[float]],   # [K][T] — matching per-turn rewards
    optimizers,                           # single optimizer OR list of optimizers
    max_grad_norm: float = 1.0,
    retain_graph: bool = False,           # ignored — kept for API compat
    eps: float = 1e-8,
) -> float:
    """One GRPO gradient step over K rollouts using episode-by-episode accumulation.

    Processes one episode at a time and calls backward() immediately, so only
    one episode's computation graph lives in GPU memory at once.  This is
    critical for large models (Qwen3-4B) where holding K graphs simultaneously
    causes OOM.

    When multiple optimizers are passed (e.g. [optimizer_a, optimizer_b]),
    a single backward accumulates gradients for all trainable params at once —
    no retain_graph needed.

    Returns the total scalar loss (detached, summed across all episodes).
    """
    if not episode_records:
        return 0.0

    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    # Episode-level mean reward (scalar per episode)
    ep_means: List[float] = [
        sum(rs) / max(len(rs), 1) for rs in episode_rewards
    ]

    # Group statistics
    K = len(ep_means)
    group_mean = sum(ep_means) / K
    group_var = sum((m - group_mean) ** 2 for m in ep_means) / max(K, 1)
    group_std = group_var ** 0.5

    for opt in optimizers:
        opt.zero_grad()

    total_loss = 0.0
    for records, ep_mean in zip(episode_records, ep_means):
        advantage = (ep_mean - group_mean) / (group_std + eps)
        ep_loss = torch.tensor(0.0)
        for record in records:
            if record.log_prob is not None:
                ep_loss = ep_loss + (-record.log_prob * float(advantage))

        if ep_loss.requires_grad:
            # backward() frees this episode's graph immediately after use
            ep_loss.backward()
            total_loss += ep_loss.item()

    all_params = [p for opt in optimizers for pg in opt.param_groups for p in pg["params"]]
    if any(p.grad is not None for p in all_params):
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        for opt in optimizers:
            opt.step()

    return total_loss


def group_stats(episode_rewards: List[List[float]]) -> dict:
    """Compute summary statistics over a group of K episodes (for logging)."""
    ep_means = [sum(rs) / max(len(rs), 1) for rs in episode_rewards]
    K = len(ep_means)
    if K == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mu = sum(ep_means) / K
    sigma = (sum((m - mu) ** 2 for m in ep_means) / K) ** 0.5
    return {"mean": mu, "std": sigma, "min": min(ep_means), "max": max(ep_means)}


def wandb_log_step(
    wandb_run,
    step: int,
    episodes,           # List[Episode]
    rewards_a: List[List[float]],
    rewards_b: List[List[float]],
    loss_a: float,
    loss_b: float,
    model,              # PeftModel — for LoRA norm tracking
    probe,              # SVDPersonaProbe — for trait name lookup
    probe_layer: int,
    use_persona_lora: bool,
    log_transcript_every: int = 50,
) -> None:
    """Log rich training diagnostics to wandb."""
    if wandb_run is None:
        return

    import torch

    stats_a = group_stats(rewards_a)
    stats_b = group_stats(rewards_b)

    # ── 1. Core scalars ──────────────────────────────────────────────────────
    log = {
        "loss/adapter_a":    loss_a,
        "loss/adapter_b":    loss_b,
        "reward_a/mean":     stats_a["mean"],
        "reward_a/std":      stats_a["std"],
        "reward_a/min":      stats_a["min"],
        "reward_a/max":      stats_a["max"],
        "reward_b/mean":     stats_b["mean"],
        "reward_b/std":      stats_b["std"],
        "reward_b/min":      stats_b["min"],
        "reward_b/max":      stats_b["max"],
        "grpo/advantage_std": stats_a["std"],  # group std = effective advantage scale
        "episode/mean_turns": sum(len(ep.records) for ep in episodes) / max(len(episodes), 1),
        "episode/mean_length_chars": sum(
            sum(len(r.action) for r in ep.records) for ep in episodes
        ) / max(sum(len(ep.records) for ep in episodes), 1),
    }

    # ── 2. Persona space per layer ───────────────────────────────────────────
    # Aggregate mean z-vector across all trainee turns in the group, per layer.
    layer_z_sums: dict = {}
    layer_z_counts: dict = {}
    for ep in episodes:
        for record in ep.records:
            if not record.probe_z_all:
                continue
            for layer_key, z_list in record.probe_z_all.items():
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
        # Cosine sim to each trait direction C[layer]
        if layer_int in probe._C:
            C = probe._C[layer_int].float()              # [N_dedup, k]
            z_norm = mean_z.norm().clamp(min=1e-8)
            C_norms = C.norm(dim=1).clamp(min=1e-8)
            sims = (C @ mean_z) / (C_norms * z_norm)    # [N_dedup]
            for slug, sim in zip(probe._slugs, sims.tolist()):
                log[f"persona/layer_{layer_key}/{slug}"] = sim

    # ── 3. LoRA B-matrix norms per layer ────────────────────────────────────
    # Tracks how much each adapter has moved from its zero initialization.
    for name, param in model.named_parameters():
        if "lora_B" not in name or not param.requires_grad:
            continue
        adapter = "adapter_a" if "adapter_a" in name else "adapter_b" if "adapter_b" in name else None
        if adapter is None:
            continue
        parts = name.split(".")
        try:
            layer_idx = next(int(p) for p in parts if p.isdigit())
        except StopIteration:
            continue
        mod = "o_proj" if "o_proj" in name else "q_proj" if "q_proj" in name else "other"
        log[f"lora_norm/{adapter}/layer_{layer_idx}/{mod}"] = param.data.norm().item()

    # ── 4. Gradient norms (read from .grad if available) ────────────────────
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

    # ── 5. Persona time series — per-turn trajectory within one sample episode ─
    # Logs a wandb.Table with one row per turn: (step, episode_idx, turn,
    # layer, slug, cosine_sim).  Gives the within-episode arc, not just the
    # cross-episode mean.  Uses only the first episode to keep table size sane.
    try:
        import wandb as _wandb
        if episodes:
            sample_ep = episodes[0]
            rows = []
            for turn_idx, record in enumerate(sample_ep.records):
                if not record.probe_z_all:
                    continue
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
                table = _wandb.Table(
                    columns=["step", "turn", "layer", "trait", "cosine_sim"],
                    data=rows,
                )
                log["persona_timeseries/episode_sample"] = table
    except Exception:
        pass

    # ── 6. Sample transcript (every N steps) ────────────────────────────────
    if step % log_transcript_every == 0 and episodes:
        ep = episodes[0]
        lines = []
        for i, r in enumerate(ep.records):
            lines.append(f"**Turn {i+1} (trainee)**")
            lines.append(f"> {r.obs[:300]}{'...' if len(r.obs)>300 else ''}")
            lines.append(f"*Action:* {r.action[:300]}{'...' if len(r.action)>300 else ''}")
            if r.probe_z and layer_key in layer_z_sums:
                top = probe.rank_traits(r.probe_z, probe_layer)[:3]
                lines.append(f"*Top traits (layer {probe_layer}):* " +
                              ", ".join(f"{s} {v:.2f}" for s, v in top))
            lines.append("")
        try:
            import wandb
            log["transcript/sample"] = wandb.Html("<br>".join(
                l.replace("**", "<b>").replace("*", "<i>") for l in lines
            ))
        except Exception:
            pass

    wandb_run.log(log, step=step)
