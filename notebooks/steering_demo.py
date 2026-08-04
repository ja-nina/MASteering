# notebooks/steering_demo.py
"""Gradio demo for token-level activation steering.

Launch:
    python notebooks/steering_demo.py            # local only
    python notebooks/steering_demo.py --share    # public Gradio link
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from PIL import Image

from notebooks.steering_core import (
    ScheduleEntry,
    best_layer,
    load_model,
    load_vectors,
    plot_probe,
    run_generation,
)

# ---------------------------------------------------------------------------
# Schedule parsing helper (also tested directly)
# ---------------------------------------------------------------------------

def _parse_schedule(rows: List[List]) -> List[ScheduleEntry]:
    """Convert Gradio Dataframe rows into ScheduleEntry objects.

    Row format: [trait, layer, start, end, coeff, mode]
    Blank or None `end` cell → ScheduleEntry.end = None (steer to end).
    """
    entries = []
    for row in rows:
        if not row or not row[0]:
            continue
        trait = str(row[0]).strip()
        layer = int(row[1])
        start = int(row[2])
        end_raw = str(row[3]).strip() if row[3] not in (None, "") else ""
        end: Optional[int] = int(end_raw) if end_raw else None
        coeff = float(row[4])
        mode = str(row[5]).strip() if row[5] else "additive"
        entries.append(ScheduleEntry(trait, layer, start, end, coeff, mode))
    return entries


# ---------------------------------------------------------------------------
# Module-level state — populated lazily in launch()
# ---------------------------------------------------------------------------

model = None
tokenizer = None
vectors: dict = {}
ALL_TRAITS: list = []


# ---------------------------------------------------------------------------
# Generate callback
# ---------------------------------------------------------------------------

def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return Image.open(buf).copy()


def generate_callback(
    prompt: str,
    schedule_df,
    probe_traits_selected: List[str],
    probe_layer: int,
    max_new_tokens: int,
    enable_thinking: bool,
):
    rows = schedule_df.values.tolist() if hasattr(schedule_df, "values") else schedule_df
    schedule = _parse_schedule(rows)

    if not probe_traits_selected:
        probe_traits_selected = list({e.trait for e in schedule})

    text, token_strings, probe_data = run_generation(
        model, tokenizer, prompt,
        schedule=schedule,
        vectors=vectors,
        probe_traits=probe_traits_selected,
        probe_layers=[probe_layer],
        max_new_tokens=int(max_new_tokens),
        enable_thinking=bool(enable_thinking),
    )

    fig = plot_probe(
        token_strings, probe_data, schedule,
        layer=probe_layer,
        traits=probe_traits_selected,
    )
    return text, _fig_to_pil(fig)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_DEFAULT_SCHEDULE = [
    ["sycophantic", 29, 0, "", 1.25, "additive"],
    ["angry",       29, 30, "", 0.8, "additive"],
]

with gr.Blocks(title="Steering Lab") as demo:
    gr.Markdown("## Activation Steering Lab — Qwen3-14B 4-bit")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="Prompt",
                value="You are a player in the Mafia game. Your role is Mafia. Describe your strategy.",
                lines=4,
            )
            schedule_table = gr.Dataframe(
                headers=["trait", "layer", "start", "end (blank=∞)", "coeff", "mode"],
                datatype=["str", "number", "number", "str", "number", "str"],
                value=_DEFAULT_SCHEDULE,
                row_count=(4, "dynamic"),
                col_count=(6, "fixed"),
                label="Steering schedule",
            )
            probe_traits_box = gr.CheckboxGroup(
                choices=[],   # populated after model loads
                value=[],
                label="Probe traits",
            )
            probe_layer_radio = gr.Radio(
                choices=[10, 20, 29],
                value=29,
                label="Probe layer",
            )
            max_tokens_slider = gr.Slider(
                minimum=64, maximum=512, step=32, value=256,
                label="Max new tokens",
            )
            thinking_checkbox = gr.Checkbox(
                value=False,
                label="Enable reasoning (<think> mode)",
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Generated text", lines=12)
            output_plot = gr.Image(label="Probe scores", type="pil")

    generate_btn.click(
        fn=generate_callback,
        inputs=[
            prompt_box, schedule_table, probe_traits_box,
            probe_layer_radio, max_tokens_slider, thinking_checkbox,
        ],
        outputs=[output_text, output_plot],
    )


# ---------------------------------------------------------------------------
# Entry point — model loads here, not at module level
# ---------------------------------------------------------------------------

def launch(share: bool = False, port: int = 7860):
    global model, tokenizer, vectors, ALL_TRAITS

    VECTORS_DIR = os.path.expandvars("${PERSONA_VECTORS_ROOT}/bf16")
    MODEL_ID = "Qwen/Qwen3-14B"

    print("Loading model…")
    model, tokenizer = load_model(MODEL_ID, bits=4)
    print("Loading vectors…")
    vectors = load_vectors(VECTORS_DIR)
    ALL_TRAITS = sorted(vectors.keys())
    print(f"Ready — {len(ALL_TRAITS)} traits available.")

    # Update probe traits checkboxes with actual trait list
    probe_traits_box.choices = ALL_TRAITS
    probe_traits_box.value = [t for t in ["sycophantic", "angry", "ethical", "trustworthiness"] if t in ALL_TRAITS]

    demo.launch(share=share, server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(share=args.share, port=args.port)
