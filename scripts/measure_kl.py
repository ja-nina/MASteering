"""Measure KL divergence between LoRA checkpoints and the base model.

Loads a no-persona LoRA checkpoint, generates responses in an IPD context,
and computes per-token KL(lora || base) averaged over responses.

Outputs a JSON summary and prints a calibration recommendation.

Usage
-----
    # Single run (auto-discovers latest checkpoint):
    python scripts/measure_kl.py \
        --model Qwen/Qwen3-4B \
        --lora-dir /scratch/.../loras_ipd_nudge_v3_nopersona/ipdNV3np_max_evil

    # Sweep across many checkpoints (glob):
    python scripts/measure_kl.py \
        --model Qwen/Qwen3-4B \
        --lora-glob "/scratch/.../loras_ipd_nudge_v3_nopersona/ipdNV3np_max_*" \
        --n-samples 20 \
        --output kl_calibration.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from testbed.training.generation_utils import STRUCTURED_FORMAT_INSTRUCTION

# ── synthetic IPD prompts (varied to get diverse generations) ─────────────────

SYSTEM = (
    "You are a strategic game player. Think carefully before acting — "
    "thorough reasoning leads to better decisions."
)

USER_TEMPLATES = [
    "Round {r}. Your opponent cooperated last round. What do you do?",
    "Round {r}. Your opponent defected last round. What do you do?",
    "Round {r}. This is the first round. What do you do?",
    "Round {r}. Scores — you: {s1}, opponent: {s2}. What do you do?",
    "Round {r}. Your opponent has defected twice in a row. What do you do?",
    "Round {r}. You've been cooperating. Opponent cooperates too. Continue?",
    "Round {r}. Final round approaching. What is your strategy?",
    "Round {r}. Mixed history. 3 cooperate, 2 defect from opponent. What now?",
]


MATH_QUESTIONS = [
    "What is 17 × 24? Show your working.",
    "A train travels 360 km in 4 hours. What is its average speed in km/h?",
]


def make_prompts(n: int, tokenizer) -> List[str]:
    prompts = []
    for i in range(n):
        tmpl = USER_TEMPLATES[i % len(USER_TEMPLATES)]
        user = tmpl.format(r=i + 1, s1=i * 3, s2=i * 2 + 1)
        msgs = [
            {"role": "system", "content": SYSTEM + STRUCTURED_FORMAT_INSTRUCTION},
            {"role": "user",   "content": user},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ))
    # append 2 math sanity checks (plain system, no game format)
    for q in MATH_QUESTIONS:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ))
    return prompts


# ── checkpoint discovery (mirrors lora_social_probe.py) ──────────────────────

def find_latest_checkpoint(lora_dir: Path) -> Path:
    checkpoints = sorted(lora_dir.glob("checkpoint_step*"), key=lambda p: p.name)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_step* in {lora_dir}")
    # actual layout: checkpoint_stepNNNNN/adapter_b/adapter_b/adapter_model.safetensors
    adapter_dir = checkpoints[-1] / "adapter_b" / "adapter_b"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter_b/adapter_b/ not found under {checkpoints[-1]}")
    return adapter_dir


# ── core KL measurement ───────────────────────────────────────────────────────

@torch.no_grad()
def measure_kl_for_checkpoint(
    base_model,
    tokenizer,
    lora_dir: Path,
    n_samples: int,
    max_new_tokens: int,
    device: str,
    temperature: float,
) -> tuple:
    """Load a LoRA adapter, generate n_samples responses, compute KL vs base.

    Returns (result_dict, restored_base_model).
    The caller MUST replace its base_model reference with the returned model —
    PeftModel.from_pretrained modifies the model in-place; unload() reverses it.
    """
    from peft import PeftModel

    adapter_path = find_latest_checkpoint(lora_dir)
    print(f"  adapter : {adapter_path}")

    lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    lora_model.eval()

    prompts = make_prompts(n_samples, tokenizer)

    per_response_kls: List[float] = []
    per_response_ntokens: List[int] = []
    per_response_ttrs: List[float] = []
    per_response_base_ppls: List[float] = []

    sample_texts:      List[str] = []  # first 3 IPD responses (LoRA)
    sample_texts_base: List[str] = []  # same prompts, base model
    math_texts:        List[str] = []  # math sanity-check responses (LoRA)
    math_texts_base:   List[str] = []  # math sanity-check responses (base)

    n_ipd = len(prompts) - len(MATH_QUESTIONS)
    log_prompt_indices = set(range(3)) | set(range(n_ipd, n_ipd + len(MATH_QUESTIONS)))

    for prompt_idx, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        ctx_len = input_ids.shape[1]

        # Generate with LoRA (adapters enabled by default)
        out = lora_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

        resp_ids = out[0, ctx_len:]  # [T]
        if resp_ids.shape[0] == 0:
            continue
        full_ids = out[0:1]          # [1, ctx+T]

        decoded = tokenizer.decode(resp_ids, skip_special_tokens=True)
        if prompt_idx >= n_ipd:
            math_texts.append(decoded)
        elif prompt_idx < 3:
            sample_texts.append(decoded)

        # For logged prompts, also generate from base for side-by-side comparison
        if prompt_idx in log_prompt_indices:
            lora_model.disable_adapters()
            base_out = lora_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            lora_model.enable_adapters()
            base_decoded = tokenizer.decode(base_out[0, ctx_len:], skip_special_tokens=True)
            if prompt_idx >= n_ipd:
                math_texts_base.append(base_decoded)
            else:
                sample_texts_base.append(base_decoded)

        # LoRA logits (adapters on — default state)
        lora_model.enable_adapters()
        lora_logits = lora_model(full_ids).logits[0]  # [ctx+T, V]

        # Base logits (adapters off)
        lora_model.disable_adapters()
        base_logits = lora_model(full_ids).logits[0]  # [ctx+T, V]
        lora_model.enable_adapters()  # restore for next iteration

        # Response-token slice: logits[t] predicts token t+1
        sl = slice(ctx_len - 1, ctx_len - 1 + resp_ids.shape[0])
        resp_lora_logits = lora_logits[sl]
        resp_base_logits = base_logits[sl]

        lora_log_probs = F.log_softmax(resp_lora_logits.float(), dim=-1)
        base_log_probs = F.log_softmax(resp_base_logits.float(), dim=-1)
        lora_probs = lora_log_probs.exp()

        # KL(lora || base) per token, then mean
        kl_per_token = (lora_probs * (lora_log_probs - base_log_probs)).sum(dim=-1)
        mean_kl = kl_per_token.mean().item()

        # Base model perplexity of the generated tokens
        resp_token_log_probs = base_log_probs[torch.arange(resp_ids.shape[0]), resp_ids]
        base_ppl = math.exp(-resp_token_log_probs.mean().item())

        # Type-token ratio (lexical diversity)
        toks = resp_ids.tolist()
        ttr = len(set(toks)) / len(toks) if toks else 0.0

        per_response_kls.append(mean_kl)
        per_response_ntokens.append(resp_ids.shape[0])
        per_response_ttrs.append(ttr)
        per_response_base_ppls.append(base_ppl)

    # Unload adapter weights and return the clean base model for the next iteration.
    # unload() reverses the in-place modification done by from_pretrained().
    restored_base = lora_model.unload()
    torch.cuda.empty_cache()

    if not per_response_kls:
        return {"lora_dir": str(lora_dir), "error": "no valid responses"}, restored_base

    mean_kl   = sum(per_response_kls) / len(per_response_kls)
    mean_ppl  = sum(per_response_base_ppls) / len(per_response_base_ppls)
    mean_ttr  = sum(per_response_ttrs) / len(per_response_ttrs)
    mean_ntok = sum(per_response_ntokens) / len(per_response_ntokens)

    # Print sample generations side-by-side (LoRA vs base)
    for idx, (lora_t, base_t) in enumerate(zip(sample_texts, sample_texts_base)):
        print(f"  [sample {idx} LoRA] {lora_t[:250].replace(chr(10), ' ')}")
        print(f"  [sample {idx} base] {base_t[:250].replace(chr(10), ' ')}")
    for idx, (q, lora_t, base_t) in enumerate(zip(MATH_QUESTIONS, math_texts, math_texts_base)):
        print(f"  [math {idx}] Q: {q}")
        print(f"  [math {idx} LoRA] {lora_t[:250].replace(chr(10), ' ')}")
        print(f"  [math {idx} base] {base_t[:250].replace(chr(10), ' ')}")

    result = {
        "lora_dir":           str(lora_dir),
        "adapter":            str(adapter_path),
        "n_samples":          len(per_response_kls),
        "mean_kl_nats":       round(mean_kl, 4),
        "std_kl_nats":        round(_std(per_response_kls), 4),
        "mean_base_ppl":      round(mean_ppl, 2),
        "mean_ttr":           round(mean_ttr, 4),
        "mean_resp_tokens":   round(mean_ntok, 1),
        "per_response_kls":   [round(v, 4) for v in per_response_kls],
        "sample_texts":       sample_texts,
        "sample_texts_base":  sample_texts_base,
        "math_texts":         math_texts,
        "math_texts_base":    math_texts_base,
    }
    return result, restored_base


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Measure KL(LoRA || base) for calibration")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--lora-dir",  type=Path, help="Single LoRA directory")
    grp.add_argument("--lora-glob", type=str,  help="Glob pattern for multiple LoRA directories")

    parser.add_argument("--model",          default="Qwen/Qwen3-4B")
    parser.add_argument("--n-samples",      type=int,   default=30,   help="Prompts per checkpoint")
    parser.add_argument("--max-new-tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--output",         type=Path,  default=None, help="Save JSON results here")
    parser.add_argument("--dtype",          default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # ── load base model once ──────────────────────────────────────────────────
    print(f"Loading base model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    base_model.eval()
    print(f"Base model loaded ({sum(p.numel() for p in base_model.parameters()) / 1e9:.2f}B params)")

    # ── collect directories to sweep ─────────────────────────────────────────
    if args.lora_dir:
        lora_dirs = [args.lora_dir]
    else:
        lora_dirs = [Path(p) for p in sorted(glob.glob(args.lora_glob)) if Path(p).is_dir()]
        print(f"Found {len(lora_dirs)} LoRA directories matching glob")

    if not lora_dirs:
        print("ERROR: no LoRA directories found", file=sys.stderr)
        sys.exit(1)

    # ── measure ──────────────────────────────────────────────────────────────
    results = []
    for i, lora_dir in enumerate(lora_dirs):
        print(f"\n[{i+1}/{len(lora_dirs)}] {lora_dir.name}")
        try:
            r, base_model = measure_kl_for_checkpoint(
                base_model=base_model,
                tokenizer=tokenizer,
                lora_dir=lora_dir,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            r = {"lora_dir": str(lora_dir), "error": str(e)}
        results.append(r)

        if "mean_kl_nats" in r:
            print(f"  KL     : {r['mean_kl_nats']:.4f} ± {r['std_kl_nats']:.4f} nats")
            print(f"  base ppl: {r['mean_base_ppl']:.2f}   TTR: {r['mean_ttr']:.3f}   "
                  f"mean tokens: {r['mean_resp_tokens']:.0f}")

    # ── summary ───────────────────────────────────────────────────────────────
    valid = [r for r in results if "mean_kl_nats" in r]
    if valid:
        kls = [r["mean_kl_nats"] for r in valid]
        ppls = [r["mean_base_ppl"] for r in valid]
        print("\n" + "═" * 60)
        print("CALIBRATION SUMMARY")
        print("═" * 60)
        print(f"  Runs measured     : {len(valid)}")
        print(f"  KL range          : {min(kls):.3f} – {max(kls):.3f} nats")
        print(f"  KL mean ± std     : {sum(kls)/len(kls):.3f} ± {_std(kls):.3f} nats")
        print(f"  Base PPL range    : {min(ppls):.1f} – {max(ppls):.1f}")
        print()
        median_kl = sorted(kls)[len(kls) // 2]
        target    = round(median_kl * 1.1, 1)   # 10% headroom above median
        print(f"  RECOMMENDED kl_target  : {target:.1f} nats")
        print(f"  RECOMMENDED initial β  : {max(0.01, round(0.5 / target, 3))}")
        print(f"  Adaptive rule: if KL > 1.5×target → β×=2; if KL < 0.5×target → β÷=2")
        print("═" * 60)

    # ── save ──────────────────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"args": vars(args) | {"lora_dir": str(args.lora_dir) if args.lora_dir else None},
                       "results": results}, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nFull results (JSON):")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
