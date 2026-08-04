# notebooks/steering_demo.py
"""Gradio demo for token-level activation steering.

Launch:
    python notebooks/steering_demo.py            # local only
    python notebooks/steering_demo.py --share    # public Gradio link

Logs are written to steering_demo.log in the repo root.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import traceback
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Logging — writes to steering_demo.log next to the repo root
# ---------------------------------------------------------------------------

_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "steering_demo.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("steering_demo")

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
    for i, row in enumerate(rows):
        if not row or not row[0]:
            log.debug("row %d: empty/None first cell, skipping", i)
            continue
        if len(row) < 6:
            log.debug("row %d: only %d columns, skipping", i, len(row))
            continue
        # Gradio sends NaN (float) for empty numeric cells — skip those rows
        trait = str(row[0]).strip()
        if not trait or trait.lower() == "nan":
            log.debug("row %d: trait is NaN/empty, skipping", i)
            continue
        try:
            layer = int(float(row[1]))
            start = int(float(row[2]))
            end_raw = str(row[3]).strip() if row[3] not in (None, "") else ""
            if end_raw.lower() in ("nan", "inf"):
                end_raw = ""
            end: Optional[int] = int(float(end_raw)) if end_raw else None
            coeff = float(row[4])
            mode = str(row[5]).strip() if row[5] else "additive"
        except (ValueError, TypeError) as exc:
            log.warning("row %d: parse error (%s), raw=%r, skipping", i, exc, row)
            continue
        log.debug("row %d: parsed → trait=%r layer=%d start=%d end=%r coeff=%s mode=%r",
                  i, trait, layer, start, end, coeff, mode)
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
    probe_layer,          # may arrive as int or str from Gradio Radio
    max_new_tokens: int,
    enable_thinking: bool,
):
    log.info("=== generate_callback invoked ===")
    log.info("  prompt[:80]  = %r", prompt[:80])
    log.info("  probe_layer  = %r (type=%s)", probe_layer, type(probe_layer).__name__)
    log.info("  max_tokens   = %s  thinking=%s", max_new_tokens, enable_thinking)
    log.info("  model loaded = %s", model is not None)
    log.info("  vectors keys = %s", sorted(vectors.keys()) if vectors else "EMPTY")

    try:
        # ── cast probe_layer to int (Gradio Radio may return str) ──────────
        probe_layer_int = int(probe_layer)

        # ── parse schedule ──────────────────────────────────────────────────
        raw_rows = schedule_df.values.tolist() if hasattr(schedule_df, "values") else schedule_df
        log.info("  raw DataFrame rows (%d total):", len(raw_rows))
        for j, r in enumerate(raw_rows):
            log.info("    [%d] %r", j, r)

        schedule = _parse_schedule(raw_rows)
        log.info("  parsed schedule (%d entries):", len(schedule))
        for e in schedule:
            log.info("    %r", e)

        # ── probe traits ────────────────────────────────────────────────────
        if not probe_traits_selected:
            # Fallback: traits from schedule, or all traits at the probe layer
            probe_traits_selected = list({e.trait for e in schedule})
            if not probe_traits_selected:
                probe_traits_selected = [
                    t for t in ALL_TRAITS if probe_layer_int in vectors.get(t, {})
                ][:6]
        log.info("  probe_traits (after fallback) = %r", probe_traits_selected)
        log.info("  probe_traits = %r", probe_traits_selected)
        log.info("  probe_layer  = %d", probe_layer_int)

        # ── generate ────────────────────────────────────────────────────────
        log.info("  calling run_generation …")
        text, token_strings, probe_data = run_generation(
            model, tokenizer, prompt,
            schedule=schedule,
            vectors=vectors,
            probe_traits=probe_traits_selected,
            probe_layers=[probe_layer_int],
            max_new_tokens=int(max_new_tokens),
            enable_thinking=bool(enable_thinking),
        )
        log.info("  generation done — %d tokens", len(token_strings))
        log.info("  probe_data layers: %s", list(probe_data.keys()))
        for pl, records in probe_data.items():
            log.info("    layer %d: %d records, sample[0]=%r", pl, len(records),
                     records[0] if records else None)

        # ── plot ────────────────────────────────────────────────────────────
        fig = plot_probe(
            token_strings, probe_data, schedule,
            layer=probe_layer_int,
            traits=probe_traits_selected,
        )
        log.info("  plot done, returning output")
        return text, _fig_to_pil(fig)

    except Exception:
        log.error("generate_callback CRASHED:\n%s", traceback.format_exc())
        # Return a plain error message so Gradio shows something
        return f"[ERROR — see steering_demo.log]\n\n{traceback.format_exc()}", None


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
                choices=[],
                value=[],
                label="Probe traits (leave blank to auto-select from schedule/vectors)",
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

    # Populate trait checkboxes when a browser connects (after launch() loads vectors)
    @demo.load(outputs=probe_traits_box)
    def _populate_traits():
        defaults = [t for t in ["sycophantic", "angry", "ethical", "trustworthiness"]
                    if t in ALL_TRAITS]
        log.info("demo.load: populating traits — choices=%s  defaults=%s", ALL_TRAITS, defaults)
        return gr.update(choices=ALL_TRAITS, value=defaults)


# ---------------------------------------------------------------------------
# Entry point — model loads here, not at module level
# ---------------------------------------------------------------------------

def launch(share: bool = False, port: int = 7860, vectors_dir: str = ""):
    global model, tokenizer, vectors, ALL_TRAITS

    MODEL_ID = "Qwen/Qwen3-14B"

    # Resolve vectors directory: CLI arg > env var > error
    if vectors_dir:
        VECTORS_DIR = vectors_dir
    else:
        raw = os.environ.get("PERSONA_VECTORS_ROOT", "")
        if raw:
            VECTORS_DIR = os.path.join(raw, "bf16")
        else:
            VECTORS_DIR = ""
    log.info("PERSONA_VECTORS_ROOT env = %r", os.environ.get("PERSONA_VECTORS_ROOT", "<not set>"))
    log.info("VECTORS_DIR resolved to: %r", VECTORS_DIR)

    if not VECTORS_DIR:
        log.error("No vectors directory! Set PERSONA_VECTORS_ROOT or pass --vectors-dir.")
        raise RuntimeError(
            "Set PERSONA_VECTORS_ROOT env var or pass --vectors-dir PATH to the script."
        )

    log.info("Loading model %s …", MODEL_ID)
    model, tokenizer = load_model(MODEL_ID, bits=4)
    log.info("Model loaded.")

    log.info("Loading vectors from %r …", VECTORS_DIR)
    vectors = load_vectors(VECTORS_DIR)
    ALL_TRAITS = sorted(vectors.keys())
    if ALL_TRAITS:
        log.info("Vectors loaded: %d traits — %s", len(ALL_TRAITS), ALL_TRAITS)
    else:
        log.error("VECTORS EMPTY after loading from %r — check path and .pt files!", VECTORS_DIR)

    # probe_traits_box is populated via demo.load() when each browser session connects
    log.info("Launching Gradio (share=%s, port=%d)", share, port)
    demo.launch(share=share, server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--vectors-dir",
        default="",
        help="Path to the bf16 vectors directory (overrides PERSONA_VECTORS_ROOT/bf16)",
    )
    args = parser.parse_args()
    launch(share=args.share, port=args.port, vectors_dir=args.vectors_dir)
