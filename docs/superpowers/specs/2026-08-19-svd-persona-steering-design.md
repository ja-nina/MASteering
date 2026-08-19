# SVD Persona Steering & Monitoring — Design Spec

**Date:** 2026-08-19  
**Repo:** MA_Environments  
**Status:** Approved for implementation

---

## Goal

Build an SVD-based persona steering and monitoring system on top of the existing
`ActivationSteering` / `PersonaProbe` infrastructure. A researcher can:

1. Specify an agent's personality as a sparse dict of named trait weights.
2. The system maps that dict to a point in the SVD persona space and injects the
   reconstructed activation vector at one or more layers (attn / mlp / both / residual).
3. During every agent turn, hidden states are projected back onto the SVD basis and the
   resulting coordinates — plus the nearest traits — are stored in the episode log.
4. A Gradio demo lets a human play any of the 8 chatty judge-free TextArena games against
   a steered agent while watching the live SVD projection after each agent turn.

---

## Background

### Existing infrastructure used

| Component | File | Role |
|---|---|---|
| `ActivationSteering` | `testbed/steering/activation.py` | Single-layer CAA vector injection; additive / adaptive / rotation modes |
| `PersonaProbe` | `testbed/probing/persona_probe.py` | Projects hidden states onto raw trait vectors per layer |
| `TransformersPolicy` | `testbed/policy/transformers_policy.py` | Registers hooks via `_HookSession`; calls both steering and probe per forward pass |
| `SteeringSpec` | `testbed/types.py` | Dataclass wiring steering config through orchestrator → policy |

### SVD persona space (from PersVecGen)

- **M** ∈ ℝ^{N×d}: CAA matrix, row i = mean-diff vector for trait i at a given layer and hook
- **SVD**: M = U Σ V^T → keep top-k right singular vectors **Vk** ∈ ℝ^{k×d}
- **k** = effective rank = round(1 / Σ pᵢ²) where pᵢ = σᵢ / Σ σⱼ; empirically 20–35 per layer for Qwen3-4B
- **C** = M @ Vk.T ∈ ℝ^{N×k}: coordinate matrix — row i is trait i's position in the k-dim persona space
- **Injection**: target point z ∈ ℝ^k → injection vector **g = Vk.T @ z** ∈ ℝ^d (projects back to model dimension, constrained to the k-dim subspace)

### Hook types (matching PersVecGen `HOOK_KEYS`)

| hook name | sublayer hooked in HF Qwen3/Llama | captures |
|---|---|---|
| `attn` | `model.layers.{N}.self_attn` | attention output delta |
| `mlp` | `model.layers.{N}.mlp` | MLP output delta |
| `residual` | `model.layers.{N}` | full residual stream (existing behavior) |
| `both` | both `self_attn` and `mlp` | simultaneous dual injection |

The SVD basis **must be built with the same hook type** that will be used at inference —
attn_delta basis vectors differ from mlp_delta basis vectors.

---

## Trait deduplication

With 53 traits many directions are near-parallel in activation space (e.g. "agreeable" and
"warm"). Running SVD on raw 53-trait matrix yields poorly interpretable PCs. Before building
the SVD basis, near-duplicate traits are merged:

1. Compute pairwise cosine similarity of all 53 CAA vectors at `ref_layer` (default 18).
2. Any pair with cosine sim > `merge_threshold` (default 0.90) shares a single representative
   vector (component-wise mean of the pair). Merge is applied greedily (highest-sim pair first).
3. All 53 original trait names remain valid in configs — merged traits map to the same z-point.
4. A `merge_map: {trait_name: canonical_name}` is stored in the basis file for transparency.
5. N_dedup ≤ 53 unique vectors remain; SVD runs on the N_dedup × d matrix.

---

## Files created / modified

```
scripts/
  build_svd_basis.py            NEW  offline basis construction

testbed/
  probing/
    svd_probe.py                NEW  SVDPersonaProbe
  steering/
    svd_steering.py             NEW  SVDSteering

notebooks/
  persona_game_demo.py          NEW  Gradio human-vs-agent demo
```

No existing files are modified. New components plug into the existing hook session and
config loading patterns.

---

## Component 1: `scripts/build_svd_basis.py`

**Purpose:** Offline script. Reads raw `*_raw.pt` trait files, deduplicates, runs SVD,
saves a single basis file per `(hook, model)` combination.

**Inputs (CLI):**
```
--std-dir     path to standard trait raw .pt files   (e.g. data/vector_extraction/persona/qwen3-4b/bf16)
--amoral-dir  path to amoral trait raw .pt files     (e.g. data/vector_extraction/persona/qwen3-4b-amoral-roleplay)
--hook        attn | mlp | residual                  (default: attn)
--model       model identifier string                (e.g. qwen3-4b)
--ref-layer   layer used for dedup similarity        (default: 18)
--merge-threshold  cosine sim threshold for merge   (default: 0.90)
--rank        int to override effective rank;        (default: auto)
--out         output .pt file path
```

**Algorithm:**
```python
# 1. Build combined CAA matrix per layer
for layer in all_layers:
    M_std   = build_caa_matrix(std_dir,   hook, layer, std_slugs)   # [N_std, d]
    M_amoral = build_caa_matrix(amoral_dir, hook, layer, amoral_slugs)  # [N_amoral, d]
    M[layer] = cat([M_std, M_amoral], dim=0)   # [53, d]

# 2. Dedup at ref_layer
sims = cosine_similarity(M[ref_layer])         # [53, 53]
merge_map = greedy_merge(sims, threshold=0.90) # {slug: canonical_slug}
slugs_dedup, M_dedup = apply_merge(merge_map, slugs, M)  # [N_dedup, d] per layer

# 3. SVD per layer
for layer in all_layers:
    U, sigma, Vt = torch.linalg.svd(M_dedup[layer].float(), full_matrices=False)
    p = sigma / sigma.sum()
    k = rank if rank is not None else round(1.0 / (p**2).sum().item())
    Vk[layer]    = Vt[:k]              # [k, d]
    C[layer]     = M_dedup[layer] @ Vk[layer].T  # [N_dedup, k]
    sigma_k[layer] = sigma[:k]

# 4. Save
torch.save({
    "hook": hook,
    "model": model,
    "slugs": slugs_dedup,          # N_dedup canonical names
    "all_slugs": all_slugs,        # all 53 original names
    "merge_map": merge_map,        # {orig_slug: canonical_slug}
    "Vk":    Vk,                   # {layer: Tensor[k, d]}
    "C":     C,                    # {layer: Tensor[N_dedup, k]}
    "sigma": sigma_k,              # {layer: Tensor[k]}
    "effective_rank": {layer: k},  # {layer: int}
}, out_path)
```

---

## Component 2: `testbed/probing/svd_probe.py`

**Class: `SVDPersonaProbe`**

Drop-in companion to `PersonaProbe`; used when `probing.mode: svd` in YAML.

**Constructor:**
```python
SVDPersonaProbe(
    basis_path: str,           # path to basis .pt file (env-var expanded)
    layers: List[int],         # layers to probe simultaneously
    hook: str = "attn",        # must match basis hook type
    layer_path_template: str = "model.layers.{}",
    window_tokens: int = 10,   # chunk size for temporal tracking
    top_k: int = 5,            # top traits to surface
)
```

**Hook registration:** one hook per layer, attached to the correct sublayer for `hook`:
- `attn` → `model.layers.{N}.self_attn`
- `mlp` → `model.layers.{N}.mlp`
- `residual` → `model.layers.{N}`

Follows exact same `make_hook() → (hooks_list, get_result_fn)` pattern as `PersonaProbe`.

**Projection per token:**
```python
h_last = output[:, -1, :].detach().float().mean(0)   # [d]
z = Vk[layer] @ h_last                                # [k]
```

**Nearest-trait lookup** (for display):
```python
# C[layer]: [N_dedup, k]  — trait coordinates in SVD space
# z: [k]
sims = C[layer] @ z / (C[layer].norm(dim=1) * z.norm() + 1e-8)  # [N_dedup]
top_traits = sorted(zip(slugs_dedup, sims), key=lambda x: -x[1])[:top_k]
```

**Output per turn** (stored in episode JSONL under `"svd_probe"`):
```json
{
  "18": {
    "z":          [0.82, -0.31, 0.14, ...],
    "top_traits": [["agreeable", 0.71], ["warm", 0.65], ...]
  },
  "27": { ... }
}
```

---

## Component 3: `testbed/steering/svd_steering.py`

**Class: `SVDSteering`**

Replaces `ActivationSteering` when `steering.default: svd` in YAML.

**Constructor:**
```python
SVDSteering(
    basis_path: str,
    per_agent: Dict[str, Dict],     # agent_id → {persona: {trait: weight}, hook, layers, coefficient, mode}
    default_config: Optional[Dict] = None,
    mode: str = "additive",         # additive | adaptive | rotation
)
```

**Persona dict → injection vector computation:**
```python
def _build_injection(self, persona: Dict[str, float], layer: int) -> torch.Tensor:
    # Build weight vector w over N_dedup canonical traits
    w = torch.zeros(len(self.slugs_dedup))
    for slug, weight in persona.items():
        canonical = self.merge_map.get(slug, slug)
        idx = self.slug_to_idx[canonical]
        w[idx] += weight
    # z = C.T @ w  (SVD point)
    z = self.C[layer].T @ w          # [k]
    # g = Vk.T @ z  (injection vector in model space)
    g = self.Vk[layer].T @ z         # [d]
    return g
```

**Hook registration:** for each layer in `layers`, registers hooks on the correct sublayer:
- `hook: attn` → one hook on `model.layers.{N}.self_attn`
- `hook: mlp` → one hook on `model.layers.{N}.mlp`
- `hook: both` → two hooks per layer (attn + mlp)
- `hook: residual` → one hook on `model.layers.{N}`

Hook function body is **identical** to `make_steering_hook()` from `activation.py` —
only the registration point changes. Reuse that function directly.

**YAML config schema:**
```yaml
steering:
  default: svd
  basis_path: ${SVD_BASIS}        # path to precomputed basis .pt
  mode: additive                  # additive | adaptive | rotation
  per_agent:
    player_0:
      hook: attn                  # attn | mlp | both | residual
      layers: [10, 18, 27]        # inject at all three simultaneously
      coefficient: 1.0
      persona:
        agreeable:      1.5
        warm:           0.0
        dominant:      -0.5
        curious:        0.0
        deceptive:      0.0
        # ... all 53 traits listed; zeros explicit
```

**`steering_spec()` return:** a list of `SteeringSpec` objects (one per layer), or a new
`SVDSteeringSpec` dataclass. Since `TransformersPolicy.act()` currently accepts a single
`SteeringSpec`, the steering integration point needs a small extension: when `steering` is
an `SVDSteering` instance, `act()` calls `steering.apply_hooks(agent_id, model)` which
returns a list of `(layer_path, hook_fn)` pairs ready for `_HookSession`. This bypasses
the single-spec limitation without breaking existing `ActivationSteering` usage.

---

## Component 4: `notebooks/persona_game_demo.py`

**Purpose:** Gradio app for human-vs-agent play in any of the 8 chatty judge-free games
with live SVD projection display after each agent turn.

**Target games:** DontSayIt-v0, SimpleNegotiation-v0, Taboo-v0, TruthAndDeception-v0,
CharacterConclave-v0, Diplomacy-v0, Negotiation-v0, SecretMafia-v0

**Layout (three-panel):**

```
┌─────────────────────┬──────────────────────────┬─────────────────────┐
│  Game transcript    │  SVD projection monitor  │  Agent controls     │
│  scrolling chat     │  horizontal bar chart    │  Game selector      │
│  history            │  showing z coords per    │  Layer selector     │
│                     │  PC, labelled with       │  Hook type          │
│  [Human text box]   │  top-2 trait names       │                     │
│  [Send]             │  Updates after each      │  Persona sliders    │
│                     │  agent turn              │  (all 53 traits,    │
│                     │                          │   -2.0 to +2.0)     │
│                     │  Layer: [dropdown]       │  [Apply] button     │
└─────────────────────┴──────────────────────────┴─────────────────────┘
```

**Data flow per turn:**
1. Human submits message → fed to TextArena env as human player action
2. TextArena returns agent observation
3. `TransformersPolicy.act()` called with `SVDPersonaProbe` (and optionally `SVDSteering`)
4. `_last_probe` dict read → z coordinates extracted for selected layer
5. Bar chart updated with z values; top-2 trait labels overlaid on bars

**Persona sliders:** all 53 trait names listed alphabetically, each a float slider from -2.0
to +2.0, default 0.0. "Apply" button recomputes the injection vectors from the current slider
state and registers updated hooks for subsequent agent turns. No model reload required.

**Startup config (CLI args to the demo script):**
```
--model       model ID (default: Qwen/Qwen3-4B)
--basis       path to SVD basis .pt file
--hook        attn | mlp | both | residual (default: attn)
--layers      comma-separated layer ints (default: 10,18,27)
--game        TextArena env ID (default: SimpleNegotiation-v0)
```

---

## Config loading integration

The existing `testbed/config.py` (or orchestrator YAML loader) is extended with two new
`mode` values:

```yaml
probing:
  mode: svd        # activates SVDPersonaProbe instead of PersonaProbe
  basis_path: ...
  hook: attn
  layers: [10, 18, 27]
  window_tokens: 10
  top_k: 5

steering:
  default: svd     # activates SVDSteering
  basis_path: ...
  mode: additive
  per_agent:
    player_0:
      hook: attn
      layers: [10, 18, 27]
      coefficient: 1.0
      persona: { ... }
```

When `mode` is absent or `trait` (existing), the original `PersonaProbe` / `ActivationSteering`
path is taken unchanged — full backward compatibility.

---

## Testing

- `tests/probing/test_svd_probe.py` — unit test with a 3-trait toy basis; verify z shape,
  top-trait ranking, chunk accumulation
- `tests/steering/test_svd_steering.py` — verify g reconstruction shape; verify `hook: both`
  registers two hooks; verify merged traits map to the same z-point
- `tests/test_build_svd_basis.py` — smoke test: build basis from fixture raw .pt files,
  check merge_map, check Vk orthonormality, check C shape

---

## Out of scope for this spec

- vLLM policy integration (SVD hooks require access to HF model internals)
- Multi-GPU / tensor-parallel inference
- Automated SVD-basis versioning or experiment tracking
- Gradio multi-user sessions
