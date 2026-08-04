import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ── Cell 1: Setup ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # make steering_core importable

from notebooks.steering_core import (
    ScheduleEntry, load_model, load_vectors, best_layer,
    run_generation, plot_probe,
)
import matplotlib.pyplot as plt

VECTORS_DIR = os.path.expandvars(
    "${PERSONA_VECTORS_ROOT}/bf16"
)
MODEL_ID = "Qwen/Qwen3-14B"

# Load once — takes ~2 min on first run
model, tokenizer = load_model(MODEL_ID, bits=4)
vectors = load_vectors(VECTORS_DIR)
print(f"Loaded {len(vectors)} trait vectors.")
"""))

# ── Cell 2: Inspect ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
print("Available traits:")
for trait in sorted(vectors.keys()):
    bl = best_layer(trait, VECTORS_DIR)
    print(f"  {trait:<35} best layer: {bl}")
"""))

# ── Cell 3: Schedule (edit this cell each experiment) ─────────────────────
cells.append(nbf.v4.new_code_cell("""\
prompt = "You are a player in the Mafia game. Your role is Mafia. Describe your strategy."

schedule = [
    ScheduleEntry("sycophantic", layer=29, start=0,  end=None, coeff=1.25, mode="additive"),
    ScheduleEntry("angry",       layer=29, start=30, end=None, coeff=0.8,  mode="additive"),
]

probe_traits = ["sycophantic", "angry", "ethical", "trustworthiness"]
probe_layers = [10, 20, 29]
enable_thinking = False
max_new_tokens = 256
"""))

# ── Cell 4: Generate ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
text, token_strings, probe_data = run_generation(
    model, tokenizer, prompt,
    schedule=schedule,
    vectors=vectors,
    probe_traits=probe_traits,
    probe_layers=probe_layers,
    max_new_tokens=max_new_tokens,
    enable_thinking=enable_thinking,
)
print(f"Generated {len(token_strings)} tokens.\\n")
print(text)
"""))

# ── Cell 5: Plot ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
for layer in probe_layers:
    fig = plot_probe(token_strings, probe_data, schedule, layer=layer, traits=probe_traits)
    plt.show()
"""))

nb.cells = cells
with open("notebooks/steering_lab.ipynb", "w") as f:
    nbf.write(nb, f)
print("Written: notebooks/steering_lab.ipynb")
