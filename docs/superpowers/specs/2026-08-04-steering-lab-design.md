# Steering Lab — Design Spec
Date: 2026-08-04

## Goal
An interactive environment for exploring token-level activation steering on Qwen3-14B (4-bit, local/server). Supports multi-vector overlapping schedules, per-token probe scoring, a hackable notebook, and a shareable Gradio demo.

---

## Files

| Path | Role |
|------|------|
| `notebooks/steering_core.py` | Shared logic — model loading, vector loading, generation, plotting |
| `notebooks/steering_lab.ipynb` | Researcher notebook — free-form exploration |
| `notebooks/steering_demo.py` | Gradio app — shareable demo via `--share` |

---

## Data structures

### `ScheduleEntry`
```python
@dataclass
class ScheduleEntry:
    trait: str          # must match a key in the loaded vectors dict
    layer: int          # integer layer index (e.g. 29)
    start: int          # first generated token to steer (0-indexed)
    end: Optional[int]  # last token (exclusive); None = steer until end of generation
    coeff: float        # steering coefficient α
    mode: str = "additive"  # "additive" | "adaptive"
```

Multiple entries may share the same layer (effects are summed) or overlap in token range.

---

## `steering_core.py` — three public functions

### `load_model(model_id, bits=4) -> (model, tokenizer)`
- Loads `model_id` (default `"Qwen/Qwen3-14B"`) with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` when `bits=4`.
- `bits=None` loads in bf16 (full precision fallback).
- Returns `(model, tokenizer)`. Model set to `eval()`.

### `load_vectors(vectors_dir) -> Dict[str, Dict[int, Tensor]]`
- Globs all `*.pt` files in `vectors_dir`.
- Each `.pt` is a PersVecGen dict `{layer_int: tensor}` or a plain tensor (loaded as `{None: tensor}`).
- Returns `{trait_name: {layer_int: tensor}}`.
- Also reads companion `*.json` files (if present) to expose `best_layer(trait)` helper — reads `sweep_scores`, returns layer int with highest `max_score - baseline`.

### `run_generation(model, tokenizer, prompt, schedule, vectors, probe_traits, probe_layers, max_new_tokens=256, enable_thinking=False) -> (text, token_strings, probe_data)`

**Hook setup:**
1. Shared `token_counter = {"t": 0}`.
2. Group `schedule` entries by `layer`. For each unique layer, build one composite steering hook.
3. Register a tick hook at the numerically lowest scheduled layer. This hook increments `token_counter["t"]` once per forward call (= once per generated token) and runs before any steering logic.
4. Each composite steering hook reads `t = token_counter["t"]`, sums `entry.coeff * vectors[entry.trait][entry.layer]` for all entries where `entry.start <= t` and `(entry.end is None or t < entry.end)`. Applies the sum via additive or adaptive formula (ported from `testbed/steering/activation.py`).
5. For each `probe_layer`, register a probe hook that appends `{trait: float}` for each trait in `probe_traits` to a per-layer list.

**Generation:**
- Applies chat template with `enable_thinking=enable_thinking`.
- When `enable_thinking=True`: uses `skip_special_tokens=False` and strips trailing EOS; keeps `<think>…</think>` block visible in output.
- When `enable_thinking=False`: `skip_special_tokens=True`.
- All hooks registered via a context manager (removed after generation).

**Returns:**
- `text`: full decoded string
- `token_strings`: list of per-token decoded strings (one per generated token)
- `probe_data`: `{layer_int: [{"trait": score, ...}, ...]}` — one dict per generated token per probed layer

### `plot_probe(token_strings, probe_data, schedule, layer, traits) -> matplotlib.Figure`
- x-axis: token index (labeled with the token string every N ticks)
- y-axis: projection score
- One line per trait in `traits` (caller passes the same list used in `run_generation`)
- Shaded translucent bands for each `ScheduleEntry` active window, labeled `{trait} α={coeff}`; `end=None` bands extend to the last token
- Horizontal dashed line at y=0
- Returns the figure (caller shows or saves it)

---

## `steering_lab.ipynb` — 5 cells

| # | Cell | Re-run? |
|---|------|---------|
| 1 | Setup: imports, `VECTORS_DIR`, `load_model()`, `load_vectors()` | Once |
| 2 | Inspect: print available traits, print `best_layer(trait)` for each | As needed |
| 3 | **Schedule**: editable `ScheduleEntry` list + `probe_traits`, `probe_layers`, `enable_thinking`, `max_new_tokens`, `prompt` | Every experiment |
| 4 | Generate: calls `run_generation()`, prints `text` | Every experiment |
| 5 | Plot: calls `plot_probe()` for each probe layer; `plt.show()` | Every experiment |

---

## `steering_demo.py` — Gradio app

Launch: `python notebooks/steering_demo.py --share`

**Startup:** `load_model()` and `load_vectors()` called once; trait list built for dropdowns.

**UI layout — two columns:**

Left (controls):
- `gr.Textbox` — prompt
- `gr.Dataframe` — schedule table with columns `[trait, layer, start, end (blank=∞), coeff, mode]`; rows editable in-browser
- `gr.CheckboxGroup` — probe traits (all available traits; default = traits in schedule)
- `gr.Radio` — probe layer `[10, 20, 29]`
- `gr.Slider` — max new tokens (64–512, default 256)
- `gr.Checkbox` — enable reasoning
- `gr.Button` — Generate

Right (outputs):
- `gr.Textbox` — generated text (with thinking block if enabled)
- `gr.Image` — probe plot (matplotlib figure saved to buffer, returned as PIL image)

**Generate callback:** parses Dataframe rows into `ScheduleEntry` list (empty `end` cell → `None`), calls `run_generation()` and `plot_probe()`, returns text + figure.

---

## Dependencies (new)
- `bitsandbytes>=0.43` — 4-bit quantization (requires CUDA + Linux/WSL2)
- `gradio>=4.0` — demo UI
- All others (`transformers`, `torch`, `matplotlib`) already present

---

## Out of scope
- Streaming token-by-token output to the Gradio UI (generation runs to completion first)
- Saving/loading named schedule presets
- Multi-prompt batch runs
