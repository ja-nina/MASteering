"""Gradio demo: human vs. steered agent in TextArena with live SVD projection.

Launch:
    python notebooks/persona_game_demo.py \\
        --model Qwen/Qwen3-4B \\
        --basis data/svd_basis/qwen3-4b_attn.pt \\
        --hook attn --layers 10,18,27 \\
        --game SimpleNegotiation-v0

Three-panel layout:
    Left:   game transcript + human text input + Send button
    Centre: SVD projection bar chart (PC-indexed bars) + top-5 trait list
    Right:  game selector, layer dropdown, hook type, persona sliders, Apply button
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import torch
import textarena as ta

from testbed.policy.transformers_policy import TransformersPolicy
from testbed.probing.svd_probe import SVDPersonaProbe
from testbed.steering.svd_steering import SVDSteering

_GAMES = [
    "DontSayIt-v0", "SimpleNegotiation-v0", "Taboo-v0",
    "TruthAndDeception-v0", "CharacterConclave-v0",
    "Diplomacy-v0", "Negotiation-v0", "SecretMafia-v0",
]

_GAME_PLAYERS = {
    "DontSayIt-v0": 2,
    "SimpleNegotiation-v0": 2,
    "Taboo-v0": 2,
    "TruthAndDeception-v0": 2,
    "CharacterConclave-v0": 3,
    "Diplomacy-v0": 3,
    "Negotiation-v0": 2,
    "SecretMafia-v0": 5,
}

_HUMAN_ID = 0

_GAME_INSTRUCTIONS = {
    "DontSayIt-v0": (
        "**Don't Say It** — 2 players. "
        "You each have a secret word the other must not say. "
        "Converse naturally and try to *trick* your opponent into saying your secret word "
        "while avoiding saying theirs. First to say the other's secret word loses."
    ),
    "SimpleNegotiation-v0": (
        "**Simple Negotiation** — 2 players. "
        "You and the agent each have private item valuations. "
        "Exchange offers by typing a proposed split (e.g. 'I propose: A=2, B=1'). "
        "Reach a deal you both accept, or walk away. Higher personal value wins."
    ),
    "Taboo-v0": (
        "**Taboo** — 2 players. "
        "One player gives clues to get the other to guess a secret word, "
        "without using any of the listed forbidden words. "
        "Type your clue or guess each turn."
    ),
    "TruthAndDeception-v0": (
        "**Truth & Deception** — 2 players. "
        "One player knows the truth; the other tries to uncover it through questions. "
        "The informed player may lie. Ask probing questions or answer strategically."
    ),
    "CharacterConclave-v0": (
        "**Character Conclave** — 3 players. "
        "Each player is assigned a secret character role. Through discussion, "
        "identify who is who before others identify you. "
        "Stay in character and vote strategically."
    ),
    "Diplomacy-v0": (
        "**Diplomacy** — 3 players. "
        "Negotiate alliances and issue orders. "
        "Coordinate or betray — only one player can win. "
        "Submit your orders and negotiations each round."
    ),
    "Negotiation-v0": (
        "**Negotiation** — 2 players. "
        "Divide a set of items between yourself and the agent. "
        "Each item has a hidden value to each party. "
        "Propose splits and counter-offer until you agree or time runs out."
    ),
    "SecretMafia-v0": (
        "**Secret Mafia** — 5 players. "
        "You are player 0. Mafia members know each other; civilians do not. "
        "During the day phase: discuss and vote to eliminate a suspect. "
        "At night: Mafia chooses a target. Civilians win by eliminating all Mafia."
    ),
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DemoState:
    """Mutable demo state. Not thread-safe; Gradio is single-threaded per session."""
    def __init__(self):
        self.env = None
        self.game_id: Optional[str] = None
        self.done = False
        self.transcript: List[str] = []
        # history: list of (turn_idx, top_trait_name, top_trait_score, top5)
        # accumulated across agent turns in the current game
        self.probe_history: List[tuple] = []
        self.policy: Optional[TransformersPolicy] = None
        self.probe: Optional[SVDPersonaProbe] = None
        self.steering: Optional[SVDSteering] = None
        self.lock = threading.Lock()


_STATE = DemoState()


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------

_SERIES_COLORS = [
    "#4a90d9", "#e05a5a", "#5ab85a", "#e0a020",
    "#9b59b6", "#1abc9c", "#e67e22", "#e91e8c",
    "#00bcd4", "#8d6e63",
]


def _make_timeseries_html(history: list, title: str = "") -> str:
    """Multi-line SVG time series: one line per trait that appears in any turn's top-k.

    history: list of (turn_idx, top_traits) where top_traits = [[slug, score], ...]
    X-axis: turn numbers; x-labels thin when dense so words don't overlap.
    """
    if not history:
        return "<p>No projection data yet.</p>"

    W, H = 460, 200
    PAD_L, PAD_R, PAD_T, PAD_B = 42, 12, 22, 44
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    n = len(history)

    # Collect all traits that ever appeared, preserving first-seen order
    seen_traits: list = []
    trait_set: set = set()
    for _, top_traits in history:
        for slug, _ in top_traits:
            if slug not in trait_set:
                seen_traits.append(slug)
                trait_set.add(slug)

    # Build per-trait score series: {slug: {turn_idx: score}}
    trait_scores: dict = {slug: {} for slug in seen_traits}
    for turn_idx, top_traits in history:
        for slug, score in top_traits:
            trait_scores[slug][turn_idx] = float(score)

    turn_ids = [t for (t, _) in history]
    all_scores = [s for d in trait_scores.values() for s in d.values()]
    y_min = min(0.0, min(all_scores))
    y_max = max(0.01, max(all_scores))
    y_range = y_max - y_min or 1.0

    def px(turn_i, score):
        x = PAD_L + (turn_i / max(n - 1, 1)) * plot_w
        y = PAD_T + plot_h - ((score - y_min) / y_range) * plot_h
        return x, y

    # Y-axis grid + labels (3 ticks)
    yticks = []
    for frac in [0.0, 0.5, 1.0]:
        val = y_min + frac * y_range
        y = PAD_T + plot_h - frac * plot_h
        yticks.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + plot_w}" y2="{y:.1f}" '
            f'stroke="#ddd" stroke-dasharray="3,3"/>'
            f'<text x="{PAD_L - 4}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="9" fill="#888">{val:.2f}</text>'
        )

    # One polyline + dots per trait
    series_els = []
    for t_idx, slug in enumerate(seen_traits):
        color = _SERIES_COLORS[t_idx % len(_SERIES_COLORS)]
        scores_map = trait_scores[slug]
        # Segments: connected run of consecutive turns where trait is present
        pts_parts = []
        seg = []
        for i, tid in enumerate(turn_ids):
            if tid in scores_map:
                seg.append((i, scores_map[tid]))
            else:
                if seg:
                    pts_parts.append(seg)
                    seg = []
        if seg:
            pts_parts.append(seg)

        for seg in pts_parts:
            if len(seg) == 1:
                # isolated point — draw as dot only
                i, sc = seg[0]
                cx, cy = px(i, sc)
                series_els.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" '
                    f'fill="{color}" stroke="white" stroke-width="1">'
                    f'<title>{slug} (turn {turn_ids[i]}): {sc:.2f}</title></circle>'
                )
            else:
                pts_str = " ".join(f"{px(i,sc)[0]:.1f},{px(i,sc)[1]:.1f}" for i, sc in seg)
                series_els.append(
                    f'<polyline points="{pts_str}" fill="none" stroke="{color}" '
                    f'stroke-width="2" stroke-linejoin="round"/>'
                )
                for i, sc in seg:
                    cx, cy = px(i, sc)
                    series_els.append(
                        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
                        f'fill="{color}" stroke="white" stroke-width="1">'
                        f'<title>{slug} (turn {turn_ids[i]}): {sc:.2f}</title></circle>'
                    )

    # X-axis turn labels — show every Nth so they don't overlap (~48 px each)
    max_labels = max(1, plot_w // 48)
    step = max(1, (n - 1) // max_labels + 1) if n > 1 else 1
    xlabels = []
    for i, (tid, top_traits) in enumerate(history):
        if i % step != 0 and i != n - 1:
            continue
        x, _ = px(i, y_min)
        # label = dominant trait name at this turn, truncated
        name = top_traits[0][0] if top_traits else str(tid)
        label = name[:9] + "…" if len(name) > 10 else name
        xlabels.append(
            f'<text x="{x:.1f}" y="{PAD_T + plot_h + 13}" text-anchor="end" '
            f'font-size="9" fill="#555" '
            f'transform="rotate(-35,{x:.1f},{PAD_T + plot_h + 13})">{label}</text>'
        )

    # Legend (right of chart, stacked)
    legend_x = PAD_L + plot_w + 6
    legend_els = []
    for t_idx, slug in enumerate(seen_traits[:10]):
        color = _SERIES_COLORS[t_idx % len(_SERIES_COLORS)]
        ly = PAD_T + t_idx * 16
        label = slug[:11] + "…" if len(slug) > 12 else slug
        legend_els.append(
            f'<rect x="{legend_x}" y="{ly}" width="10" height="10" fill="{color}" rx="2"/>'
            f'<text x="{legend_x + 13}" y="{ly + 9}" font-size="9" fill="#444">{label}</text>'
        )

    title_el = (
        f'<text x="{(PAD_L + PAD_L + plot_w) // 2}" y="14" text-anchor="middle" '
        f'font-size="11" font-weight="600" fill="#333">{title}</text>'
    ) if title else ""

    svg_w = W + 110  # extra room for legend
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{H}" '
        f'style="max-width:100%;font-family:sans-serif;overflow:visible">'
        + title_el
        + "".join(yticks)
        + "".join(series_els)
        + "".join(xlabels)
        + "".join(legend_els)
        + f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + plot_h}" stroke="#aaa"/>'
        + f'<line x1="{PAD_L}" y1="{PAD_T + plot_h}" x2="{PAD_L + plot_w}" y2="{PAD_T + plot_h}" stroke="#aaa"/>'
        + "</svg>"
    )
    return svg


# ---------------------------------------------------------------------------
# Game control
# ---------------------------------------------------------------------------

def _update_persona(persona_values: Dict[str, float], layers: List[int], hook: str) -> None:
    """Recompute injection vectors from slider state (no model reload)."""
    if _STATE.steering is None or _STATE.policy is None:
        return
    # Drive all non-human slots with the same steered policy
    num_players = _GAME_PLAYERS.get(_STATE.game_id, 2) if _STATE.game_id else 2
    _STATE.steering._per_agent = {
        str(pid): {
            "hook": hook,
            "layers": layers,
            "coefficient": 1.0,
            "persona": persona_values,
        }
        for pid in range(1, num_players)
    }


def _auto_play_agents_until_human(probe_layer: Optional[int]) -> Tuple[str, str]:
    """Drive all agent turns until human's turn or game end (must be called within lock)."""
    chart_html = "<p>Play a turn to see projection.</p>"
    while not _STATE.done:
        current_player, obs_str = _STATE.env.get_observation()
        if current_player == _HUMAN_ID:
            break
        # Non-human slot: auto-generate
        system_prompt = "You are a competitive game player. Respond concisely."
        action, _ = _STATE.policy.act(
            system_prompt=system_prompt,
            user_prompt=str(obs_str),
            agent_id=str(current_player),
            steering=None,  # SVDSteering handled via apply_hooks inside act()
        )
        _STATE.transcript.append(f"[Agent {current_player}] {action}")
        done, _info = _STATE.env.step(action)
        _STATE.done = done
        if done:
            _STATE.transcript.append("[Game over]")
            chart_html = "<p>Game over.</p>"
            break
        # Update time series if probe_layer requested
        if probe_layer is not None and _STATE.policy._last_probe:
            layer_data = _STATE.policy._last_probe.get(str(probe_layer), {})
            top_traits = layer_data.get("top_traits", [])
            if top_traits:
                turn_idx = len(_STATE.probe_history) + 1
                _STATE.probe_history.append((turn_idx, top_traits))
                chart_html = _make_timeseries_html(
                    _STATE.probe_history, title=f"SVD persona monitor (layer {probe_layer})"
                )
    return "<br>".join(_STATE.transcript), chart_html


def _start_game(game_id: str, persona_values: Dict[str, float],
                hook: str, layers: List[int],
                probe_layer: Optional[int] = None) -> Tuple[str, str]:
    """Initialise or re-initialise the TextArena env and agent."""
    global _STATE
    with _STATE.lock:
        num_players = _GAME_PLAYERS.get(game_id, 2)
        env = ta.make(game_id)
        env.reset(num_players=num_players)
        _STATE.env = env
        _STATE.game_id = game_id
        _STATE.done = False
        _STATE.transcript = [f"[Game started: {game_id}]"]
        _STATE.probe_history = []

        if _STATE.policy is not None and _STATE.steering is not None:
            _update_persona(persona_values, layers, hook)

        current_player, obs_str = env.get_observation()
        _STATE.transcript.append(f"[Env] {obs_str}")

        if current_player != _HUMAN_ID and not _STATE.done:
            transcript_html, chart_html = _auto_play_agents_until_human(probe_layer=probe_layer)
        else:
            transcript_html = "<br>".join(_STATE.transcript)
            chart_html = "<p>Play a turn to see projection.</p>"

    return transcript_html, chart_html


def _human_turn(human_text: str, probe_layer: int) -> Tuple[str, str]:
    """Process one human action then auto-play agents until human's turn again."""
    global _STATE
    with _STATE.lock:
        if _STATE.env is None or _STATE.done:
            return "Start a game first.", "<p>No game running.</p>"

        # 1. Step the env with the human's action (step takes a plain string)
        done, _info = _STATE.env.step(human_text)
        _STATE.transcript.append(f"[You] {human_text}")
        _STATE.done = done

        if done:
            _STATE.transcript.append("[Game over]")
            return "<br>".join(_STATE.transcript), "<p>Game over.</p>"

        # 2. Auto-play agents until it's the human's turn again
        transcript_html, chart_html = _auto_play_agents_until_human(probe_layer)

    return transcript_html, chart_html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="Qwen/Qwen3-4B")
    parser.add_argument("--basis",  required=True)
    parser.add_argument("--hook",   default="attn",
                        choices=["attn", "mlp", "both", "residual"])
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer ints; defaults to all layers in basis")
    parser.add_argument("--game",   default="SimpleNegotiation-v0")
    parser.add_argument("--share",  action="store_true")
    args = parser.parse_args(argv)

    # Load basis to extract trait names for sliders
    basis = torch.load(args.basis, map_location="cpu", weights_only=False)
    all_slugs: List[str] = basis["all_slugs"]

    layers: Optional[List[int]] = (
        [int(x) for x in args.layers.split(",")] if args.layers
        else sorted(basis["Vk"].keys())  # all layers in basis by default
    )

    # Build probe (monitor)
    probe = SVDPersonaProbe(
        basis_path=args.basis,
        layers=layers,
        hook=args.hook,
        top_k=7,
    )

    # Build steering (starts with all zeros)
    initial_persona = {slug: 0.0 for slug in all_slugs}
    steering = SVDSteering(
        basis_path=args.basis,
        per_agent={
            "1": {
                "hook": args.hook,
                "layers": layers,
                "coefficient": 1.0,
                "persona": initial_persona,
            }
        },
    )

    # Build policy (load model once)
    policy = TransformersPolicy(
        model_id=args.model,
        steering=steering,
        probe=probe,
    )

    _STATE.policy = policy
    _STATE.probe = probe
    _STATE.steering = steering

    def _instructions_html(game_id: str) -> str:
        text = _GAME_INSTRUCTIONS.get(game_id, "Select a game to see instructions.")
        # convert **bold** markers to <strong>
        import re
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        return f'<div style="padding:8px;border-left:3px solid #888;font-size:0.9em">{text}</div>'

    # Build Gradio UI
    with gr.Blocks(title="Persona Game Demo") as demo:
        gr.Markdown("## SVD Persona Steering — Human vs. Agent")

        with gr.Row():
            # ── Left: transcript ─────────────────────────────────────────────
            with gr.Column(scale=2):
                instructions = gr.HTML(
                    label="Game instructions",
                    value=_instructions_html(args.game),
                )
                transcript = gr.HTML(label="Transcript", value="<p>Select a game and click Start.</p>")
                human_input = gr.Textbox(label="Your move", placeholder="Type your action…")
                send_btn = gr.Button("Send")

            # ── Centre: SVD chart ────────────────────────────────────────────
            with gr.Column(scale=2):
                chart = gr.HTML(label="SVD Projection", value="<p>No data yet.</p>")
                probe_layer = gr.Dropdown(
                    choices=[str(l) for l in layers],
                    value=str(layers[0]),
                    label="Display layer",
                )

            # ── Right: controls ──────────────────────────────────────────────
            with gr.Column(scale=1):
                game_sel = gr.Dropdown(choices=_GAMES, value=args.game, label="Game")
                hook_sel = gr.Dropdown(
                    choices=["attn", "mlp", "both", "residual"],
                    value=args.hook, label="Hook type",
                )
                start_btn = gr.Button("Start / Restart", variant="primary")
                gr.Markdown("### Persona sliders")
                sliders = {
                    slug: gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label=slug)
                    for slug in sorted(all_slugs)
                }
                apply_btn = gr.Button("Apply persona")

        # ── Event handlers ────────────────────────────────────────────────────

        def _on_game_change(game_id):
            return _instructions_html(game_id)

        def _on_start(game_id, layer_str, hook_type, *slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, hook_type)
            t, c = _start_game(game_id, persona, hook_type, layers, probe_layer=int(layer_str))
            return _instructions_html(game_id), t, c

        def _on_send(text, layer_str, hook_type, *slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, hook_type)
            t, c = _human_turn(text, int(layer_str))
            return t, c, ""

        def _on_apply(*slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, args.hook)
            return gr.update()

        slider_list = [sliders[s] for s in sorted(all_slugs)]

        game_sel.change(fn=_on_game_change, inputs=[game_sel], outputs=[instructions])

        start_btn.click(
            fn=_on_start,
            inputs=[game_sel, probe_layer, hook_sel] + slider_list,
            outputs=[instructions, transcript, chart],
        )
        send_btn.click(
            fn=_on_send,
            inputs=[human_input, probe_layer, hook_sel] + slider_list,
            outputs=[transcript, chart, human_input],
        )
        apply_btn.click(fn=_on_apply, inputs=slider_list, outputs=[chart])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
