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
        self.last_z: Optional[List[float]] = None
        self.last_top_traits: List[List] = []
        self.policy: Optional[TransformersPolicy] = None
        self.probe: Optional[SVDPersonaProbe] = None
        self.steering: Optional[SVDSteering] = None
        self.lock = threading.Lock()


_STATE = DemoState()


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------

def _make_bar_chart_html(z: List[float], top_traits: List[List], title: str = "") -> str:
    """Return HTML: PC-indexed bar chart for SVD z-coordinates + top-trait text list."""
    if not z:
        return "<p>No projection data yet.</p>"
    max_abs = max(abs(v) for v in z) or 1.0
    bars = []
    for i, val in enumerate(z):
        pct_pos = val / max_abs  # [-1, 1]
        label = f"PC{i}"
        color = "#4a90d9" if val >= 0 else "#e05a5a"
        width_px = int(abs(pct_pos) * 80)
        bars.append(
            f'<div style="display:flex;align-items:center;gap:4px;margin:2px 0">'
            f'<span style="font-size:11px;width:90px;text-align:right;color:#888">{label}</span>'
            f'<div style="width:160px;display:flex;justify-content:{"flex-end" if val<0 else "flex-start"}">'
            f'<div style="width:{width_px}px;height:14px;background:{color};border-radius:2px"></div>'
            f'</div>'
            f'<span style="font-size:10px;color:#666">{val:.2f}</span>'
            f'</div>'
        )
    # Top-k nearest traits as a numbered text list below the bars
    trait_items = "".join(
        f'<li style="font-size:11px">{rank}. {slug} ({score:.2f})</li>'
        for rank, (slug, score) in enumerate(top_traits[:5], start=1)
    )
    trait_list = (
        f'<div style="margin-top:8px"><b style="font-size:11px">Top traits:</b>'
        f'<ol style="margin:2px 0;padding-left:18px">{trait_items}</ol></div>'
        if trait_items else ""
    )
    return (
        f'<div style="font-size:12px;font-weight:600;margin-bottom:8px">{title}</div>'
        + "".join(bars)
        + trait_list
    )


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
        # Update chart if probe_layer requested
        if probe_layer is not None and _STATE.policy._last_probe:
            layer_data = _STATE.policy._last_probe.get(str(probe_layer), {})
            z = layer_data.get("z", None)
            top_traits = layer_data.get("top_traits", [])
            if z:
                _STATE.last_z = z
                _STATE.last_top_traits = top_traits
                chart_html = _make_bar_chart_html(
                    z, top_traits, title=f"SVD projection (layer {probe_layer})"
                )
    return "<br>".join(_STATE.transcript), chart_html


def _start_game(game_id: str, persona_values: Dict[str, float],
                hook: str, layers: List[int]) -> Tuple[str, str]:
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
        _STATE.last_z = None
        _STATE.last_top_traits = []

        if _STATE.policy is not None and _STATE.steering is not None:
            _update_persona(persona_values, layers, hook)

        current_player, obs_str = env.get_observation()
        _STATE.transcript.append(f"[Env] {obs_str}")

        if current_player != _HUMAN_ID and not _STATE.done:
            transcript_html, chart_html = _auto_play_agents_until_human(probe_layer=None)
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

        def _on_start(game_id, hook_type, *slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, hook_type)
            t, c = _start_game(game_id, persona, hook_type, layers)
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
            inputs=[game_sel, hook_sel] + slider_list,
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
