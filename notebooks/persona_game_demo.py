"""Gradio demo: human vs. steered agent in TextArena with live SVD projection.

Launch:
    python notebooks/persona_game_demo.py \\
        --model Qwen/Qwen3-4B \\
        --basis data/svd_basis/qwen3-4b_attn.pt \\
        --hook attn --layers 10,18,27 \\
        --game SimpleNegotiation-v0

Three-panel layout:
    Left:   game transcript + human text input + Send button
    Centre: SVD projection bar chart (z coordinates, top-2 trait labels per bar)
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
_HUMAN_ID = 0
_AGENT_ID = 1


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DemoState:
    """Mutable demo state. Not thread-safe; Gradio is single-threaded per session."""
    def __init__(self):
        self.env = None
        self.obs = None
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
    """Return an HTML bar chart for SVD z-coordinates."""
    if not z:
        return "<p>No projection data yet.</p>"
    k = len(z)
    bar_w = max(300, k * 24)
    max_abs = max(abs(v) for v in z) or 1.0
    bars = []
    for i, val in enumerate(z):
        pct_pos = val / max_abs  # [-1, 1]
        label = top_traits[i][0] if i < len(top_traits) else f"PC{i}"
        color = "#4a90d9" if val >= 0 else "#e05a5a"
        width_px = int(abs(pct_pos) * 80)
        margin_px = 80 - width_px if val < 0 else 80
        bars.append(
            f'<div style="display:flex;align-items:center;gap:4px;margin:2px 0">'
            f'<span style="font-size:11px;width:90px;text-align:right;color:#888">{label}</span>'
            f'<div style="width:160px;display:flex;justify-content:{"flex-end" if val<0 else "flex-start"}">'
            f'<div style="width:{width_px}px;height:14px;background:{color};border-radius:2px"></div>'
            f'</div>'
            f'<span style="font-size:10px;color:#666">{val:.2f}</span>'
            f'</div>'
        )
    return (
        f'<div style="font-size:12px;font-weight:600;margin-bottom:8px">{title}</div>'
        + "".join(bars)
    )


# ---------------------------------------------------------------------------
# Game control
# ---------------------------------------------------------------------------

def _start_game(game_id: str, persona_values: Dict[str, float],
                hook: str, layers: List[int]) -> Tuple[str, str]:
    """Initialise or re-initialise the TextArena env and agent."""
    global _STATE
    with _STATE.lock:
        env = ta.make(game_id)
        obs, info = env.reset(num_players=2)
        _STATE.env = env
        _STATE.obs = obs
        _STATE.done = False
        _STATE.transcript = [f"[Game started: {game_id}]"]
        _STATE.last_z = None
        _STATE.last_top_traits = []

        # Build SVD steering from current sliders
        if _STATE.policy is not None and _STATE.steering is not None:
            _update_persona(persona_values, layers, hook)

        obs_text = obs.get(_HUMAN_ID, obs.get(str(_HUMAN_ID), ""))
        _STATE.transcript.append(f"[Env] {obs_text}")
        transcript_html = "<br>".join(_STATE.transcript)
        chart_html = "<p>Play a turn to see projection.</p>"
    return transcript_html, chart_html


def _update_persona(persona_values: Dict[str, float], layers: List[int], hook: str) -> None:
    """Recompute injection vectors from slider state (no model reload)."""
    if _STATE.steering is None or _STATE.policy is None:
        return
    _STATE.steering._per_agent = {
        str(_AGENT_ID): {
            "hook": hook,
            "layers": layers,
            "coefficient": 1.0,
            "persona": persona_values,
        }
    }


def _human_turn(human_text: str, probe_layer: int) -> Tuple[str, str]:
    """Process one human action and one agent response."""
    global _STATE
    with _STATE.lock:
        if _STATE.env is None or _STATE.done:
            return "Start a game first.", "<p>No game running.</p>"

        # 1. Step the env with human action
        obs, rewards, done, info = _STATE.env.step({_HUMAN_ID: human_text})
        _STATE.transcript.append(f"[You] {human_text}")

        if done:
            _STATE.done = True
            _STATE.transcript.append(f"[Game over] {rewards}")
            return "<br>".join(_STATE.transcript), "<p>Game over.</p>"

        # 2. Agent turn
        agent_obs = obs.get(_AGENT_ID, obs.get(str(_AGENT_ID), ""))
        system_prompt = "You are a competitive game player. Respond concisely."
        action, _ = _STATE.policy.act(
            system_prompt=system_prompt,
            user_prompt=str(agent_obs),
            agent_id=str(_AGENT_ID),
            steering=None,  # SVDSteering handled via self.steering.apply_hooks
        )
        _STATE.transcript.append(f"[Agent] {action}")

        # 3. Step env with agent action
        obs, rewards, done, info = _STATE.env.step({_AGENT_ID: action})
        _STATE.done = done

        if done:
            _STATE.transcript.append(f"[Game over] {rewards}")

        # 4. Extract SVD projection from probe
        z = None
        top_traits: List[List] = []
        if _STATE.policy._last_probe:
            layer_data = _STATE.policy._last_probe.get(str(probe_layer), {})
            z = layer_data.get("z", None)
            top_traits = layer_data.get("top_traits", [])
        _STATE.last_z = z
        _STATE.last_top_traits = top_traits

        chart_html = _make_bar_chart_html(
            z or [], top_traits, title=f"SVD projection (layer {probe_layer})"
        )
        transcript_html = "<br>".join(_STATE.transcript)
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
    parser.add_argument("--layers", default="10,18,27")
    parser.add_argument("--game",   default="SimpleNegotiation-v0")
    parser.add_argument("--share",  action="store_true")
    args = parser.parse_args(argv)

    layers = [int(x) for x in args.layers.split(",")]

    # Load basis to extract trait names for sliders
    basis = torch.load(args.basis, map_location="cpu", weights_only=False)
    all_slugs: List[str] = basis["all_slugs"]

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
            str(_AGENT_ID): {
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

    # Build Gradio UI
    with gr.Blocks(title="Persona Game Demo") as demo:
        gr.Markdown("## SVD Persona Steering — Human vs. Agent")

        with gr.Row():
            # ── Left: transcript ─────────────────────────────────────────────
            with gr.Column(scale=2):
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

        def _on_start(game_id, hook_type, *slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, hook_type)
            t, c = _start_game(game_id, persona, hook_type, layers)
            return t, c

        def _on_send(text, layer_str, *slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, args.hook)
            t, c = _human_turn(text, int(layer_str))
            return t, c, ""

        def _on_apply(*slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, args.hook)
            return gr.update()

        slider_list = [sliders[s] for s in sorted(all_slugs)]

        start_btn.click(
            fn=_on_start,
            inputs=[game_sel, hook_sel] + slider_list,
            outputs=[transcript, chart],
        )
        send_btn.click(
            fn=_on_send,
            inputs=[human_input, probe_layer] + slider_list,
            outputs=[transcript, chart, human_input],
        )
        apply_btn.click(fn=_on_apply, inputs=slider_list, outputs=[chart])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
