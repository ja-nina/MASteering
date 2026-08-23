"""
Ablate persona LoRA vs task LoRA using saved checkpoints.

Generates N samples under four conditions and (optionally) scores each with
the SVD persona probe so you can see which adapter drives the behaviour:

  base            — base model, no adapters
  adapter_a only  — persona LoRA  (SVD-constrained, interpretable axes)
  adapter_b only  — task LoRA     (free weights, unconstrained)
  both            — adapter_a + adapter_b active together

Checkpoint layout expected (matches train_lora.py output):
  <ckpt>/adapter_a/   PEFT adapter_a weights
  <ckpt>/adapter_b/   PEFT adapter_b weights
  <ckpt>/             tokenizer (falls back to --model id if absent)

Usage:
    python scripts/ablate_loras.py \\
        --checkpoint outputs/lora/nothink/persona_ipd/checkpoint_step00500 \\
        --probe-basis data/svd_basis/qwen3-4b-residual.pt

    # Specify your own prompt:
    python scripts/ablate_loras.py \\
        --checkpoint outputs/lora/nothink/persona_ipd/final \\
        --user-prompt "Round 3. Both players cooperated last round. Your move?" \\
        --n 2 --max-new-tokens 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from testbed.training.generation_utils import STRUCTURED_FORMAT_INSTRUCTION


# ─── defaults ────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM = (
    "You are a strategic game player. Think carefully before acting — "
    "thorough reasoning leads to better decisions."
)

DEFAULT_USER = (
    "[GAME] Welcome to the Iterated Prisoner's Dilemma.\n"
    "You are Player 1. In each round both players simultaneously choose "
    "to Cooperate or Defect.\n"
    "  Both cooperate → each scores 3 pts\n"
    "  Both defect    → each scores 1 pt\n"
    "  One defects    → defector 5 pts, cooperator 0 pts\n\n"
    "This is Round 1. Please reply with '[Cooperate]' or '[Defect]'."
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _set_adapters(model, names: List[str]) -> None:
    """Activate exactly the named adapters, disable everything else."""
    for module in model.modules():
        if hasattr(module, "set_adapter"):
            module.set_adapter(names)


def _build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system + STRUCTURED_FORMAT_INSTRUCTION},
        {"role": "user",   "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


@torch.no_grad()
def _generate(model, tokenizer, system: str, user: str,
              max_new_tokens: int, device: str) -> str:
    prompt = _build_prompt(tokenizer, system, user)
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    gen_ids = out[0, ids["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


@torch.no_grad()
def _probe_text(
    model,
    tokenizer,
    text: str,
    probe,          # SVDPersonaProbe
    device: str,
) -> Dict:
    """Run a single forward pass over `text` with SVD probe hooks, return scores."""
    hooks_spec, get_result = probe.make_hook()

    # Resolve module paths and register hooks
    handles = []
    for path, hook_fn in hooks_spec:
        module = model
        for part in path.split("."):
            module = getattr(module, part)
        handles.append(module.register_forward_hook(hook_fn))

    try:
        ids = tokenizer(text, return_tensors="pt").to(device)
        # Run with adapters disabled — we probe what the text activates
        # in the base model, independent of which adapter generated it.
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                model(**ids)
        else:
            model(**ids)
    finally:
        for h in handles:
            h.remove()

    return get_result()


def _fmt_probe(scores: Dict, probe, layer: int) -> str:
    ld = scores.get(str(layer), {})
    z = ld.get("z")
    if not z:
        return "  [probe] no z vectors"
    top = probe.rank_traits(z, layer)
    lines = ["  [probe top traits]"]
    for slug, sim in top[:5]:
        bar = "█" * int(abs(sim) * 20)
        sign = "+" if sim >= 0 else "-"
        lines.append(f"    {slug:20s} {sign}{abs(sim):.3f}  {bar}")
    return "\n".join(lines)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Ablate persona vs task LoRA")
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint dir (contains adapter_a/ and/or adapter_b/)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B",
                    help="Base model id or local path")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--user-prompt",   default=None)
    ap.add_argument("--n", type=int, default=2,
                    help="Samples per condition")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-basis", default=None,
                    help="Path to qwen3-4b-residual.pt for trait scoring")
    ap.add_argument("--probe-layer", type=int, default=35,
                    help="Which layer to display trait rankings for")
    ap.add_argument("--probe-hook", default="residual",
                    help="Hook type matching the basis (residual | attn | mlp)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    has_a = (ckpt / "adapter_a").exists()
    has_b = (ckpt / "adapter_b").exists()

    if not has_a and not has_b:
        sys.exit(
            f"[ablate] Neither adapter_a/ nor adapter_b/ found under {ckpt}.\n"
            f"         Check --checkpoint points to a step or final checkpoint dir."
        )

    system = args.system_prompt or DEFAULT_SYSTEM
    user   = args.user_prompt   or DEFAULT_USER
    device = args.device

    # ── Load tokenizer ────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_dir = str(ckpt) if (ckpt / "tokenizer_config.json").exists() else args.model
    print(f"[ablate] tokenizer  ← {tok_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"[ablate] base model ← {args.model}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ── Attach adapters ───────────────────────────────────────────────────────
    from peft import PeftModel

    peft_model: Optional[PeftModel] = None

    if has_a:
        print(f"[ablate] adapter_a  ← {ckpt / 'adapter_a'}", flush=True)
        peft_model = PeftModel.from_pretrained(
            base_model, str(ckpt / "adapter_a"), adapter_name="adapter_a",
        )

    if has_b:
        print(f"[ablate] adapter_b  ← {ckpt / 'adapter_b'}", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                base_model, str(ckpt / "adapter_b"), adapter_name="adapter_b",
            )
        else:
            peft_model.load_adapter(str(ckpt / "adapter_b"), adapter_name="adapter_b")

    model = (peft_model if peft_model is not None else base_model).to(device)
    model.eval()

    # ── Optional probe ────────────────────────────────────────────────────────
    probe = None
    if args.probe_basis:
        print(f"[ablate] probe      ← {args.probe_basis}", flush=True)
        from testbed.probing.svd_probe import SVDPersonaProbe
        probe = SVDPersonaProbe(
            basis_path=args.probe_basis,
            hook=args.probe_hook,
        )

    # ── Build ablation conditions ─────────────────────────────────────────────
    #  Each entry: (label, adapter_names | None)
    #  None  → use disable_adapter() context
    #  list  → call _set_adapters(model, list)
    conditions: List[Tuple[str, Optional[List[str]]]] = [
        ("base  (no adapters)", None),
    ]
    if has_a:
        conditions.append(("adapter_a only  [persona LoRA — SVD-constrained]", ["adapter_a"]))
    if has_b:
        conditions.append(("adapter_b only  [task LoRA — free weights]", ["adapter_b"]))
    if has_a and has_b:
        conditions.append(("both  [adapter_a + adapter_b]", ["adapter_a", "adapter_b"]))

    # ── Run ───────────────────────────────────────────────────────────────────
    W = 72
    print(f"\n{'═' * W}")
    print("ABLATION PROMPT")
    print(f"  system : {system[:80]}{'…' if len(system) > 80 else ''}")
    print(f"  user   : {user[:100]}{'…' if len(user) > 100 else ''}")
    print(f"{'═' * W}\n")

    for label, adapter_names in conditions:
        print(f"\n{'╔' + '═' * (W - 2) + '╗'}")
        print(f"║  {label:<{W - 4}}║")
        print(f"{'╚' + '═' * (W - 2) + '╝'}")

        if peft_model is not None:
            if adapter_names is None:
                ctx = peft_model.disable_adapter()
                ctx.__enter__()
            else:
                ctx = None
                _set_adapters(peft_model, adapter_names)

        for i in range(args.n):
            print(f"\n─── sample {i + 1} / {args.n} ───")
            text = _generate(model, tokenizer, system, user,
                             args.max_new_tokens, device)
            print(text)

            if probe is not None:
                scores = _probe_text(model, tokenizer, text, probe, device)
                print(_fmt_probe(scores, probe, args.probe_layer))

        if peft_model is not None and adapter_names is None:
            ctx.__exit__(None, None, None)

    print(f"\n{'═' * W}")
    print("[ablate] done.")


if __name__ == "__main__":
    main()
