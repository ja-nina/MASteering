"""
Analyze what a trained LoRA has learned in terms of SVD persona directions.

Usage:
    python scripts/analyze_lora.py \
        --lora outputs/lora/persona_run_001/final \
        --basis data/svd_basis/qwen3-4b-attn.pt \
        --layer 18 \
        --prompts data/reference_prompts.txt \
        --out outputs/lora/persona_run_001/analysis.html
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_probe(basis_path, layer, hook="attn"):
    from testbed.probing.svd_probe import SVDPersonaProbe
    probe = SVDPersonaProbe(basis_path=basis_path, layers=[layer], hook=hook, top_k=20)
    return probe


def run_prompts_get_z(model, tokenizer, probe, prompts, layer, device):
    """Run each prompt through the model; return one z-vector per prompt.

    A fresh set of hooks is created per prompt so that the running-mean
    accumulator in SVDPersonaProbe is reset between inputs and each entry
    in the returned list corresponds to exactly one prompt.
    """
    from testbed.policy.transformers_policy import _HookSession
    zs = []
    for prompt in prompts:
        # Fresh make_hook() call resets the state accumulator.
        hooks, get_result = probe.make_hook()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with _HookSession(model, hooks):
            with torch.no_grad():
                model(**inputs)
        result = get_result()
        z_list = result.get(str(layer), {}).get("z", [])
        if z_list:
            zs.append(torch.tensor(z_list, dtype=torch.float32))
    return zs


def activation_shift_per_trait(z_base_list, z_lora_list, basis_path, layer):
    """For each dedup trait, compute mean cosine-sim shift: lora - base."""
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


def make_html(trait_deltas, title):
    """Simple horizontal bar chart HTML."""
    sorted_td = sorted(trait_deltas, key=lambda x: -abs(x[1]))[:20]
    max_abs = max(abs(v) for _, v in sorted_td) or 1.0
    bars = []
    for slug, delta in sorted_td:
        color = "#4a90d9" if delta >= 0 else "#e05a5a"
        w = int(abs(delta) / max_abs * 200)
        bars.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
            f'<span style="width:140px;text-align:right;font-size:11px">{slug}</span>'
            f'<div style="width:{w}px;height:14px;background:{color};border-radius:2px"></div>'
            f'<span style="font-size:10px;color:#666">{delta:+.3f}</span></div>'
        )
    body = "\n".join(bars)
    return (f"<h3>{title}</h3>"
            f"<p>Delta = LoRA cosine sim - base cosine sim to each trait direction</p>"
            f"{body}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora",    required=True, help="Path to saved LoRA adapter")
    parser.add_argument("--basis",   required=True)
    parser.add_argument("--layer",   type=int, default=18)
    parser.add_argument("--hook",    default="attn")
    parser.add_argument("--prompts", required=True, help="Text file with one prompt per line")
    parser.add_argument("--model",   default=None, help="Base model ID (read from adapter if omitted)")
    parser.add_argument("--out",     default="lora_analysis.html")
    args = parser.parse_args()

    prompts = Path(args.prompts).read_text(encoding="utf-8").splitlines()
    prompts = [p.strip() for p in prompts if p.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    peft_cfg = PeftConfig.from_pretrained(args.lora)
    base_id = args.model or peft_cfg.base_model_name_or_path

    print(f"Loading base model {base_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                                        device_map="auto")
    device = next(base_model.parameters()).device

    probe = build_probe(args.basis, args.layer, args.hook)

    print("Running reference prompts through base model...")
    z_base = run_prompts_get_z(base_model, tokenizer, probe, prompts, args.layer, device)

    print("Loading LoRA and running prompts...")
    lora_model = PeftModel.from_pretrained(base_model, args.lora)
    lora_model.merge_and_unload()   # merge for clean forward pass
    z_lora = run_prompts_get_z(lora_model, tokenizer, probe, prompts, args.layer, device)

    trait_deltas = activation_shift_per_trait(z_base, z_lora, args.basis, args.layer)
    html_body = make_html(
        trait_deltas,
        f"LoRA activation shift - layer {args.layer} ({args.hook})"
    )

    full_html = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset=\"utf-8\"><title>LoRA Analysis</title>\n"
        "<style>body{font-family:sans-serif;max-width:600px;margin:40px auto;padding:0 16px}</style>\n"
        f"</head><body>{html_body}</body></html>"
    )

    Path(args.out).write_text(full_html, encoding="utf-8")
    print(f"\nSaved analysis -> {args.out}")


if __name__ == "__main__":
    main()
