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


def _log_probs_per_token(model, full_ids: torch.Tensor, input_len: int) -> torch.Tensor:
    """Per-token log probs for the generated sequence. Returns [gen_len] tensor."""
    import torch.nn.functional as F
    gen_len = full_ids.shape[1] - input_len
    logits = model(full_ids).logits                           # [1, T, vocab]
    gen_logits = logits[0, input_len - 1: input_len - 1 + gen_len, :]
    log_probs_matrix = F.log_softmax(gen_logits, dim=-1)
    gen_ids = full_ids[0, input_len:]
    return log_probs_matrix[torch.arange(gen_len, device=full_ids.device), gen_ids]  # [gen_len]


def _compute_log_prob(model, full_ids: torch.Tensor, input_len: int) -> torch.Tensor:
    """Scalar sum of per-token log probs (requires_grad=True)."""
    return _log_probs_per_token(model, full_ids, input_len).sum()


def _action_token_offset(tokenizer, full_ids: torch.Tensor, input_len: int) -> int:
    """Token index (relative to input_len) where <action> begins.

    Decode the generated tokens, find the '<action>' char offset, then
    re-encode the strategy prefix to count its tokens.  Returns 0 when
    '<action>' is absent so the full generation is penalised.
    """
    gen_ids = full_ids[0, input_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    pos = gen_text.find("<action>")
    if pos == -1:
        return 0
    prefix_ids = tokenizer.encode(gen_text[:pos], add_special_tokens=False)
    return min(len(prefix_ids), len(gen_ids))


def grpo_step(
    episode: Episode,
    optimizers,
    max_grad_norm: float = 1.0,
    eps: float = 1e-8,
    zero_grad: bool = True,
    do_step: bool = True,
    model=None,
    device: str = "cuda",
    kl_coef: float = 0.0,
    tokenizer=None,
) -> float:
    """Accumulate GRPO gradients for one episode; optionally step the optimizers.

    For each TurnGroup:
      - (optionally) recompute log_probs with gradient tracking
      - normalise K rewards → K advantages
      - accumulate loss: Σ_k  −log_prob_k × advantage_k
      - backward() immediately → frees this turn's K graphs

    model/device: when provided, log_probs are computed inside this function
                  one turn group at a time, so only K graphs are live at once.
                  This avoids the OOM caused by front-loading all N_turns×K
                  forward passes before any backward.  When None, record.log_prob
                  must already be filled (legacy path).

    zero_grad: call opt.zero_grad() before accumulating.
    do_step:   call clip_grad_norm + opt.step() after all backwards.

    kl_coef: per-token KL penalty weight (Shao et al. 2024 GRPO formulation).
        Adds β * (log π - log π_ref) / T_action to the loss for each record,
        where T_action is the number of action tokens (after <action> tag).
        When tokenizer is provided the penalty applies ONLY to action tokens —
        the strategy section (<strategy>…</strategy>) is unconstrained, letting
        the model freely develop reasoning while keeping game decisions on-policy.
        When tokenizer is None the full generation is penalised (legacy behaviour).
        Reference log-probs are computed by temporarily disabling all LoRA
        adapters on `model` (requires a PEFT model; noop when kl_coef=0).
        β=0.02–0.04 works well for persona rewards in [-1, 1].

    Returns total scalar loss (summed across all turn groups).
    """
    if not episode.turn_groups:
        return 0.0

    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]

    if zero_grad:
        for opt in optimizers:
            opt.zero_grad()

    total_loss    = 0.0
    total_kl_loss = 0.0

    for tg in episode.turn_groups:
        K = len(tg.rewards)
        if K < 2:
            tg.advantages = [0.0] * K
            continue

        mu  = sum(tg.rewards) / K
        std = (sum((r - mu) ** 2 for r in tg.rewards) / K) ** 0.5
        advantages = [(r - mu) / (std + eps) for r in tg.rewards]
        tg.advantages = advantages

        # Recompute log_probs for this turn group only — keeps K graphs live,
        # not K×N_turns.  Falls back to pre-filled record.log_prob if no model.
        tg_loss    = torch.tensor(0.0)
        tg_kl_loss = torch.tensor(0.0)
        for record, adv in zip(tg.records, advantages):
            full_ids    = None
            gen_len     = 0
            lp_per_tok  = None   # [gen_len] tensor with grad — reused by KL block
            if model is not None and record.full_ids is not None:
                full_ids = record.full_ids.to(device)
                gen_len  = full_ids.shape[1] - record.input_len
                if gen_len > 0:
                    lp_per_tok      = _log_probs_per_token(model, full_ids, record.input_len)
                    record.log_prob = lp_per_tok.sum()
                else:
                    record.log_prob = None
            if record.log_prob is not None:
                pg_loss = -record.log_prob * float(adv)

                # Token-level KL penalty (GRPO formulation, Shao et al. 2024):
                #   β * (log π_θ − log π_ref) / T_action
                # When tokenizer is provided the penalty is restricted to action
                # tokens only (<action>…</action>), leaving the strategy section
                # free to diverge.  Without tokenizer the full generation is penalised.
                # Reference = base model with all LoRA adapters disabled.
                kl_loss = torch.tensor(0.0)
                if kl_coef > 0.0 and lp_per_tok is not None and gen_len > 0:
                    try:
                        with torch.no_grad(), model.disable_adapter():
                            lp_ref = _log_probs_per_token(
                                model, full_ids, record.input_len
                            )
                        offset = (
                            _action_token_offset(tokenizer, full_ids, record.input_len)
                            if tokenizer is not None else 0
                        )
                        n_act = gen_len - offset
                        if n_act > 0:
                            kl_loss = kl_coef * (
                                lp_per_tok[offset:] - lp_ref.detach()[offset:]
                            ).sum() / n_act
                    except Exception:
                        pass   # disable_adapter unavailable (non-PEFT model)

                tg_loss    = tg_loss    + pg_loss + kl_loss
                tg_kl_loss = tg_kl_loss + kl_loss

        if tg_loss.requires_grad:
            tg_loss.backward()   # frees this turn group's K graphs immediately
            total_loss    += tg_loss.item()
            total_kl_loss += tg_kl_loss.item()

    if do_step:
        all_params = [p for opt in optimizers for pg in opt.param_groups for p in pg["params"]]
        if any(p.grad is not None for p in all_params):
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
            for opt in optimizers:
                opt.step()

    return total_loss, total_kl_loss


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
    kl_penalty: float = 0.0,
) -> None:
    """Log rich training diagnostics to wandb."""
    if wandb_run is None:
        return

    import torch

    import wandb as _wandb

    stats = episode_stats(episode)

    # ── 1. Core scalars ──────────────────────────────────────────────────────
    n_turns = len(episode.turn_groups)
    log = {
        "loss":               loss,
        "kl_penalty":         kl_penalty,
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

    # Raw cosine-sim (pre-sign, pre-alpha, mean across layers) — sanity check that
    # the probe is measuring what we think it is. Should track reward closely when
    # sign=+1 and alpha=1; diverges when format failures (-1.0) dominate.
    all_raw = [r.raw_cos_sim for tg in episode.turn_groups
               for r in tg.records if r.raw_cos_sim is not None]
    if all_raw:
        log["reward/raw_cos_sim_mean"] = sum(all_raw) / len(all_raw)
        log["reward/raw_cos_sim_min"]  = min(all_raw)
        log["reward/raw_cos_sim_max"]  = max(all_raw)
        log["reward/raw_cos_sim_std"]  = (
            (sum((x - log["reward/raw_cos_sim_mean"]) ** 2 for x in all_raw) / len(all_raw)) ** 0.5
        )

    # ── 1b. Per-turn reward breakdown + opponent decision rates ─────────────
    # With K=8 candidates, opp_coop_rate is continuous in {0, 1/8, ..., 1}.
    # This directly measures behavioral influence, not just persona cosine-sim.
    turn_table = _wandb.Table(columns=["turn", "reward_mean", "reward_best", "reward_worst",
                                       "top_trait", "top_trait_score",
                                       "opp_coop_rate", "opp_defect_rate", "opp_invalid_rate",
                                       "opp_response_best_k", "opp_response_len_best_k",
                                       "opp_response_len_mean"])
    all_coop_rates: List[float] = []
    all_opp_resp_lens: List[float] = []
    for t_idx, tg in enumerate(episode.turn_groups):
        if not tg.rewards:
            continue
        r_mean  = sum(tg.rewards) / len(tg.rewards)
        r_best  = max(tg.rewards)
        r_worst = min(tg.rewards)
        best_k  = tg.rewards.index(r_best)
        opp_z   = tg.records[best_k].probe_z_opponent

        # Opponent decision rates across K candidates
        decisions = [r.opp_decision or "" for r in tg.records]
        K = len(decisions)
        n_coop    = sum(1 for d in decisions if "cooperat" in d)
        n_defect  = sum(1 for d in decisions if "defect"   in d)
        n_invalid = K - n_coop - n_defect
        opp_coop_rate    = n_coop    / K if K else float("nan")
        opp_defect_rate  = n_defect  / K if K else float("nan")
        opp_invalid_rate = n_invalid / K if K else float("nan")
        if K > 0:
            all_coop_rates.append(opp_coop_rate)
        top_trait, top_score = "", float("nan")
        if opp_z and probe is not None:
            try:
                ranked = probe.rank_traits(opp_z, probe_layer)
                if ranked:
                    top_trait, top_score = ranked[0][0], float(ranked[0][1])
            except Exception:
                pass

        # Opponent counterfactual response lengths
        resp_lens = [len(r.opp_response) for r in tg.records if r.opp_response]
        opp_resp_best = tg.records[best_k].opp_response or ""
        opp_resp_len_best = len(opp_resp_best)
        opp_resp_len_mean = sum(resp_lens) / len(resp_lens) if resp_lens else float("nan")
        if resp_lens:
            all_opp_resp_lens.extend(resp_lens)

        turn_table.add_data(t_idx, r_mean, r_best, r_worst, top_trait, top_score,
                            opp_coop_rate, opp_defect_rate, opp_invalid_rate,
                            opp_resp_best, opp_resp_len_best, opp_resp_len_mean)
    log["episode/turn_rewards"] = turn_table

    # Episode-level mean opponent coop rate across all turns (continuous, K=8)
    if all_coop_rates:
        log["opponent/coop_rate"]   = sum(all_coop_rates) / len(all_coop_rates)
        log["opponent/defect_rate"] = 1.0 - log["opponent/coop_rate"]
    if all_opp_resp_lens:
        log["opponent/cf_response_len_mean"] = sum(all_opp_resp_lens) / len(all_opp_resp_lens)
        log["opponent/cf_response_len_min"]  = min(all_opp_resp_lens)
        log["opponent/cf_response_len_max"]  = max(all_opp_resp_lens)
    if rolling_mean is not None:
        log["reward/rolling_mean_10"] = rolling_mean
    if step_time is not None:
        log["train/step_time_s"] = step_time
    if game_outcome is not None:
        log["episode/trainee_game_reward"] = game_outcome.get("trainee", float("nan"))
        log["episode/opponent_game_reward"] = game_outcome.get("opponent", float("nan"))
        decision = game_outcome.get("decision")
        if decision is not None:
            log["episode/decision"] = decision
            log["episode/cooperated"] = 1.0 if decision == "cooperate" else (
                0.0 if decision == "defect" else float("nan"))

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
    if episode.episode_log or episode.turn_groups:
        try:
            import wandb as _wandb

            def _esc(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            # System prompt header — show trainee's (nudge) prompt if different
            _sp_main = episode.system_prompt or ""
            _sp_trainee = episode.trainee_system_prompt
            sp_section = (
                f"<div class='sysprompt'>"
                f"<div class='sp-label'>SYSTEM PROMPT (opponent / base)</div>"
                f"<div class='sp-body'>{_esc(_sp_main)}</div>"
            )
            if _sp_trainee is not None and _sp_trainee != _sp_main:
                sp_section += (
                    f"<div class='sp-label' style='margin-top:6px'>SYSTEM PROMPT (trainee / nudge)</div>"
                    f"<div class='sp-body'>{_esc(_sp_trainee)}</div>"
                )
            sp_section += "</div>"

            n_trainee = len(episode.turn_groups)
            html_parts = [
                "<style>"
                "body{font-family:monospace;font-size:13px;padding:8px;background:#1a1a1a;color:#ddd}"
                ".sysprompt{border:1px solid #555;border-radius:4px;margin:0 0 10px;padding:8px;background:#252525}"
                ".sp-label{font-weight:bold;font-size:10px;color:#888;margin-bottom:3px;letter-spacing:.05em}"
                ".sp-body{white-space:pre-wrap;color:#aaa;font-size:11px}"
                ".turn{border:1px solid #444;border-radius:4px;margin:6px 0;padding:8px}"
                ".turn.trainee{border-color:#3a5a3a}"
                ".turn.opponent{border-color:#3a3a5a}"
                ".turn-header{font-weight:bold;margin-bottom:6px;font-size:11px}"
                ".turn.trainee .turn-header{color:#6f6}"
                ".turn.opponent .turn-header{color:#88f}"
                ".section-label{font-size:10px;color:#666;margin-bottom:2px;letter-spacing:.05em}"
                ".obs{color:#888;white-space:pre-wrap;margin-bottom:6px;font-size:11px;border-left:2px solid #444;padding-left:6px}"
                ".action{white-space:pre-wrap;color:#eee;border-left:2px solid #555;padding-left:6px}"
                ".meta{margin-top:6px;font-size:11px;color:#6cf}"
                ".reward-pos{color:#4f4}"
                "</style>"
                "<h3 style='margin:0 0 8px;color:#ccc'>Rollout transcript — "
                f"step {step} &nbsp; ({n_trainee} trainee turns, "
                f"{len(episode.episode_log)} total)</h3>"
            ]
            html_parts.append(sp_section)

            # Build a lookup: episode_log index → TurnGroup (for trainee reward/adv)
            trainee_log_idx = 0  # walks through turn_groups
            for log_entry in episode.episode_log:
                player = log_entry["player"]
                obs_txt = _esc(log_entry["obs"])
                action_txt = _esc(log_entry["action"])
                if player == "trainee":
                    tg = (episode.turn_groups[trainee_log_idx]
                          if trainee_log_idx < len(episode.turn_groups) else None)
                    trainee_log_idx += 1
                    reward = tg.rewards[0] if tg else float("nan")
                    best_r = max(tg.rewards) if tg else float("nan")
                    adv_str = (f"adv={tg.advantages[0]:+.3f}" if tg and tg.advantages else "")
                    top_trait_str = ""
                    if tg and tg.records[0].probe_z_opponent:
                        try:
                            top = probe.rank_traits(tg.records[0].probe_z_opponent, probe_layer)
                            if top:
                                top_trait_str = f"  top_trait={top[0][0]}:{top[0][1]:+.2f}"
                        except Exception:
                            pass
                    r_cls = "reward-pos" if reward > 0 else ""
                    # Best-k counterfactual opponent response (the one that drove the reward)
                    best_k_idx = tg.rewards.index(max(tg.rewards)) if tg else 0
                    cf_opp_resp = (tg.records[best_k_idx].opp_response or "") if tg else ""
                    cf_opp_len  = len(cf_opp_resp)
                    cf_decision = (tg.records[best_k_idx].opp_decision or "?") if tg else "?"
                    cf_block = (
                        f"<div class='section-label' style='margin-top:8px;color:#88f'>"
                        f"OPPONENT COUNTERFACTUAL RESPONSE (best-k, {cf_opp_len} chars, decision={cf_decision})</div>"
                        f"<div class='action' style='color:#bbf;border-color:#446'>{_esc(cf_opp_resp)}</div>"
                    ) if cf_opp_resp else ""
                    html_parts.append(
                        f"<div class='turn trainee'>"
                        f"<div class='turn-header'>TRAINEE</div>"
                        f"<div class='section-label'>OBSERVATION (full prompt context)</div>"
                        f"<div class='obs'>{obs_txt}</div>"
                        f"<div class='section-label'>RESPONSE (full, including strategy)</div>"
                        f"<div class='action'>{action_txt}</div>"
                        f"{cf_block}"
                        f"<div class='meta'>"
                        f"<span class='{r_cls}'>r={reward:+.4f}</span> "
                        f"best={best_r:+.4f}  {adv_str}{top_trait_str}"
                        f"</div></div>"
                    )
                else:
                    html_parts.append(
                        f"<div class='turn opponent'>"
                        f"<div class='turn-header'>OPPONENT</div>"
                        f"<div class='section-label'>OBSERVATION (full prompt context)</div>"
                        f"<div class='obs'>{obs_txt}</div>"
                        f"<div class='section-label'>RESPONSE (full, including strategy)</div>"
                        f"<div class='action'>{action_txt}</div>"
                        f"</div>"
                    )
            log["rollout/transcript"] = _wandb.Html("".join(html_parts))
        except Exception:
            pass

    # ── 8. Detailed Table — every N steps (for offline filtering) ────────────
    if step % log_transcript_every == 0 and episode.turn_groups:
        try:
            import wandb as _wandb
            _sp_main = episode.system_prompt or ""
            _sp_trainee = episode.trainee_system_prompt or _sp_main
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
                    opp_resp = record.opp_response or ""
                    raw_cs = record.raw_cos_sim
                    rows.append([
                        turn_idx, k,
                        _sp_trainee,
                        record.obs,
                        record.action,
                        len(record.action),
                        round(reward, 4),
                        round(adv, 4) if adv == adv else float("nan"),
                        round(raw_cs, 4) if raw_cs is not None else float("nan"),
                        top_trait,
                        round(top_sim, 4) if top_sim == top_sim else float("nan"),
                        opp_resp,
                        len(opp_resp),
                        record.opp_decision or "",
                    ])
            if rows:
                log["rollout/transcript_table"] = _wandb.Table(
                    columns=["turn", "candidate", "system_prompt",
                             "observation", "action", "action_len_chars",
                             "reward", "advantage", "raw_cos_sim",
                             "top_trait", "top_sim",
                             "opp_cf_response", "opp_cf_response_len_chars",
                             "opp_decision"],
                    data=rows,
                )
        except Exception:
            pass

    wandb_run.log(log, step=step)
