"""
Analyze what trained LoRA adapters have learned in terms of SVD persona directions.

Loads both adapter_a (maximiser) and adapter_b (minimiser) from their saved
directories, runs reference prompts through base / adapter_a / adapter_b, and
produces an HTML page with side-by-side per-trait cosine-sim delta bar charts.

Usage:
    python scripts/analyze_lora.py \
        --lora-a outputs/lora/dual_run_001/final/adapter_a \
        --lora-b outputs/lora/dual_run_001/final/adapter_b \
        --basis  data/svd_basis/qwen3-4b-attn.pt \
        --layer  18 \
        --prompts data/reference_prompts.txt \
        --out    outputs/lora/dual_run_001/analysis.html
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_probe(basis_path: str, layer: int, hook: str = "attn"):
    from testbed.probing.svd_probe import SVDPersonaProbe
    return SVDPersonaProbe(basis_path=basis_path, layers=[layer], hook=hook, top_k=20)


def run_prompts_get_z(model, tokenizer, probe, prompts, layer, device):
    """Run each prompt through the model; return one z-vector per prompt.

    A fresh make_hook() call per prompt resets the accumulator so each entry
    in the returned list corresponds to exactly one input (not a cumulative mean).
    """
    from testbed.policy.transformers_policy import _HookSession
    zs = []
    for prompt in prompts:
        hooks, get_result = probe.make_hook()   # fresh accumulator state
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with _HookSession(model, hooks):
            with torch.no_grad():
                model(**inputs)
        result = get_result()
        z_list = result.get(str(layer), {}).get("z", [])
        if z_list:
            zs.append(torch.tensor(z_list, dtype=torch.float32))
    return zs


def activation_shift_per_trait(z_base_list, z_lora_list, basis_path: str, layer: int):
    """Per-trait cosine-sim delta: mean_lora - mean_base."""
    basis = torch.load(basis_path, map_location="cpu", weights_only=False)
    C = basis["C"][layer].float()          # [N_dedup, k]
    slugs = basis["slugs"]

    def mean_cosine_to_traits(zs):
        if not zs:
            return torch.zeros(len(slugs))
        Z = torch.stack(zs)               # [n, k]
        Z_norm = Z.norm(dim=1, keepdim=True).clamp(min=1e-8)
        C_norm = C.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sims = (Z / Z_norm) @ (C / C_norm).T   # [n, N_dedup]
        return sims.mean(0)                     # [N_dedup]

    base_sims = mean_cosine_to_traits(z_base_list)
    lora_sims = mean_cosine_to_traits(z_lora_list)
    delta = lora_sims - base_sims
    return list(zip(slugs, delta.tolist()))


def _bar_chart_html(trait_deltas, title: str, top_n: int = 20) -> str:
    """Horizontal bar chart showing the top-N traits by absolute delta."""
    sorted_td = sorted(trait_deltas, key=lambda x: -abs(x[1]))[:top_n]
    max_abs = max(abs(v) for _, v in sorted_td) or 1.0
    rows = []
    for slug, delta in sorted_td:
        color = "#4a90d9" if delta >= 0 else "#e05a5a"
        w = int(abs(delta) / max_abs * 200)
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
            f'<span style="width:140px;text-align:right;font-size:11px">{slug}</span>'
            f'<div style="width:{w}px;height:14px;background:{color};border-radius:2px"></div>'
            f'<span style="font-size:10px;color:#666">{delta:+.3f}</span></div>'
        )
    body = "\n".join(rows)
    return (
        f'<div style="flex:1;min-width:280px">'
        f'<h3 style="margin-top:0">{title}</h3>'
        f'<p style="font-size:11px;color:#888">Delta = LoRA - base cosine sim per trait</p>'
        f'{body}'
        f'</div>'
    )


def make_html(delta_a, delta_b, layer: int, hook: str) -> str:
    chart_a = _bar_chart_html(delta_a, "Adapter A (maximiser)")
    chart_b = _bar_chart_html(delta_b, "Adapter B (minimiser)")
    return (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset=\"utf-8\"><title>LoRA Analysis</title>\n"
        "<style>"
        "body{font-family:sans-serif;max-width:1000px;margin:40px auto;padding:0 16px}"
        ".row{display:flex;gap:40px;flex-wrap:wrap}"
        "</style></head><body>"
        f"<h2>LoRA Dual-Adapter Activation Shift &mdash; layer {layer} ({hook})</h2>"
        f'<div class="row">{chart_a}{chart_b}</div>'
        "</body></html>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-a",  required=True, help="Path to adapter_a checkpoint dir")
    parser.add_argument("--lora-b",  required=True, help="Path to adapter_b checkpoint dir")
    parser.add_argument("--basis",   required=True)
    parser.add_argument("--layer",   type=int, default=18)
    parser.add_argument("--hook",    default="attn")
    parser.add_argument("--prompts", required=True, help="Text file, one prompt per line")
    parser.add_argument("--model",   default=None,
                        help="Base model ID (read from adapter config if omitted)")
    parser.add_argument("--out",     default="lora_analysis.html")
    args = parser.parse_args()

    prompts = Path(args.prompts).read_text(encoding="utf-8").splitlines()
    prompts = [p.strip() for p in prompts if p.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    peft_cfg = PeftConfig.from_pretrained(args.lora_a)
    base_id = args.model or peft_cfg.base_model_name_or_path

    print(f"Loading base model {base_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                        device_map="auto")
    device = next(base_model.parameters()).device

    probe = build_probe(args.basis, args.layer, args.hook)

    # Load both adapters into a single PeftModel so we can switch between them.
    print("Loading adapter_a...")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_a,
                                            adapter_name="adapter_a")
    print("Loading adapter_b...")
    peft_model.load_adapter(args.lora_b, adapter_name="adapter_b")

    # Base model run (all adapters disabled).
    print("Running reference prompts through base model...")
    with peft_model.disable_adapter():
        z_base = run_prompts_get_z(peft_model, tokenizer, probe, prompts,
                                   args.layer, device)

    # Adapter A run.
    print("Running prompts through adapter_a (maximiser)...")
    peft_model.set_adapter("adapter_a")
    z_lora_a = run_prompts_get_z(peft_model, tokenizer, probe, prompts,
                                  args.layer, device)

    # Adapter B run.
    print("Running prompts through adapter_b (minimiser)...")
    peft_model.set_adapter("adapter_b")
    z_lora_b = run_prompts_get_z(peft_model, tokenizer, probe, prompts,
                                  args.layer, device)

    delta_a = activation_shift_per_trait(z_base, z_lora_a, args.basis, args.layer)
    delta_b = activation_shift_per_trait(z_base, z_lora_b, args.basis, args.layer)

    html = make_html(delta_a, delta_b, args.layer, args.hook)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"\nSaved analysis -> {args.out}")


if __name__ == "__main__":
    main()
