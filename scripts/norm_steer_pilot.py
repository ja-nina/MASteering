"""Pilot (2026-11-19): does norm-preserving steering hurt reasoning less than plain additive steering?

A lone agent's round-1 answer to SuperGPQA 'hard' questions (the qa_debate prompt, as debate_acc at N = 1), the 2B
in non-thinking mode, HF transformers on one GPU. Per trait and coefficient, five conditions on the same questions:
    none           unsteered
    add            h + c*v at the trait's layer (vLLM-Hook's add_vector: PersVecFast's smoke F6 checks this hook)
    add_random     the same with a random direction of the same norm (testbed.steering.random_control)
    norm           h + c*v, then rescaled per token back to ||h||  (the direction changes, the size does not)
    norm_random    the same with the random direction
Also logs ||c*v|| / ||h|| per token at the layer (median over the prompt tokens): if it is small, rescaling
changes little and the pilot can only come out flat.

    python scripts/norm_steer_pilot.py --vectors-dir <PersVecFast out>/persona/qwen3.5-2b/bf16 \
        --cell assertive:1 --cell sycophantic:1 --n 40 --out ~/norm_pilot.jsonl

Differences from the games, on purpose (pilot only): HF sampling with the model card's temperature / top_p / top_k
but no presence penalty (HF has none), max_new_tokens 4096 instead of 15360 (cut answers count as wrong, the same
for every condition), one round only. Compare conditions with each other here, not with debate_acc's numbers.
Resumes: (condition, trait, coefficient, question) already in --out are skipped.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CONDITIONS = ("none", "add", "add_random", "norm", "norm_random")
AGENT = "player_0"


def steer_hidden(h: torch.Tensor, add: torch.Tensor, norm_preserving: bool) -> torch.Tensor:
    """h + add, or h + add rescaled per token (last dim) to the norm of h."""
    out = h + add.to(h.dtype)
    if norm_preserving:
        out = out * (h.float().norm(dim=-1, keepdim=True) / out.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(h.dtype)
    return out


def raw_obs(item) -> dict:
    """The qa_debate observation of a lone agent in round 1 of 3 (testbed/envs/symbolic/qa_debate.py)."""
    return {"agent_id": AGENT, "round_index": 0, "num_rounds": 3, "num_players": 1, "dataset": "supergpqa",
            "question": item["question"], "options": list(item.get("options") or []), "history": [], "show_chars": 1500}


def score(completion: str, obs: dict, answer: str):
    """(parsed letter or None, correct) with the game's own parser."""
    from testbed.parsers.symbolic.qa_debate import QADebateParser
    res = QADebateParser().parse(completion, obs, AGENT, None)
    got = res.value["answer"] if hasattr(res, "value") else None
    return got, got is not None and got == answer


def done_keys(path: Path):
    keys = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r["condition"], r["trait"], r["coef"], r["question"]))
    return keys


def report(path: Path):
    rows = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows[(r["trait"], r["coef"], r["condition"])].append(r)
    by_q = {k: {r["question"]: r for r in v} for k, v in rows.items()}
    print(f"\n{'trait':14}{'c':>5} {'condition':12}{'qs':>4}{'acc':>7}{'d none':>8}{'+-SE':>7}{'tokens':>8}{'cut':>7}"
          f"{'unpars':>8}{'|cv|/|h|':>10}")
    for trait, coef in sorted({k[:2] for k in rows if k[0] != "-"}):
        base = by_q.get(("-", 0.0, "none"), {})
        for cond in CONDITIONS:
            key = ("-", 0.0, "none") if cond == "none" else (trait, coef, cond)
            v = rows.get(key)
            if not v:
                continue
            d = [r["correct"] - base[r["question"]]["correct"] for r in v if r["question"] in base]
            m = sum(d) / len(d) if d else float("nan")
            se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5 if len(d) > 1 else float("nan")
            ratio = [r["ratio"] for r in v if r.get("ratio") is not None]
            print(f"{trait:14}{coef:>+5g} {cond:12}{len(v):>4}{sum(r['correct'] for r in v) / len(v):>7.3f}"
                  f"{m:>+8.3f}{se:>7.3f}{sum(r['n_tokens'] for r in v) / len(v):>8.0f}"
                  f"{100 * sum(r['finish'] == 'length' for r in v) / len(v):>6.1f}%"
                  f"{100 * sum(r['parsed'] is None for r in v) / len(v):>7.1f}%"
                  + (f"{sorted(ratio)[len(ratio) // 2]:>10.3f}" if ratio else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors-dir", required=True)
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--cell", action="append", required=True, help="trait:coefficient, e.g. assertive:1")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--start", type=int, default=0, help="first question index (debate_acc plays 0-199)")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if a.report_only:
        return report(out)

    from persvecfast.models import MODELS
    from persvecfast.smoke import decoder_layers, load_hf
    from transformers import AutoTokenizer

    from testbed.envs.symbolic.qa_debate import questions
    from testbed.mia_engine import template
    from testbed.renderers.symbolic.qa_debate import QADebateRenderer
    from testbed.steering import RANDOM_CONTROL_SEED, VectorStore, random_control

    spec = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(spec.hf_repo)
    tok.padding_side = "left"
    hf = load_hf(spec.hf_repo).eval()
    layers = decoder_layers(hf)
    hidden = hf.config.hidden_size
    print(f"{a.model}: {len(layers)} decoder blocks, hidden {hidden}, enable_thinking={spec.enable_thinking}", flush=True)
    store = VectorStore(a.vectors_dir, cache_dir=out.parent / "_norm_pilot_vectors")

    items = list(questions("supergpqa"))[a.start:a.start + a.n]
    renderer = QADebateRenderer()
    obs = [raw_obs(it) for it in items]
    prompts = [template(tok, spec, renderer.system_prompt(AGENT, o), renderer.render(o, AGENT, None), 0) for o in obs]
    print("first prompt:\n" + prompts[0][:1500], flush=True)

    jobs = [("none", "-", 0.0, None, None, False)]
    for cell in a.cell:
        trait, coef = cell.split(":")
        coef = float(coef)
        layer = store.best_layer(trait)
        v = store.vectors(trait)[layer].float()
        assert v.shape == (hidden,), f"{trait}: vector shape {tuple(v.shape)}, model hidden {hidden}"
        rnd = random_control(v, RANDOM_CONTROL_SEED + layer)
        print(f"{trait}: layer {layer} of {len(layers)}, ||v|| = {v.norm():.2f}, c = {coef:+g}", flush=True)
        jobs += [("add", trait, coef, layer, v, False), ("add_random", trait, coef, layer, rnd, False),
                 ("norm", trait, coef, layer, v, True), ("norm_random", trait, coef, layer, rnd, True)]

    done = done_keys(out)
    with open(out, "a", encoding="utf-8") as fh:
        for cond, trait, coef, layer, vec, norm in jobs:
            todo = [i for i in range(len(items)) if (cond, trait, coef, a.start + i) not in done]
            if not todo:
                continue
            t0 = time.time()
            for b in range(0, len(todo), a.batch):
                idx = todo[b:b + a.batch]
                ratios = []
                handle = None
                if vec is not None:
                    add = (coef * vec).cuda()

                    def hook(m, i, o, add=add, norm=norm, ratios=ratios):
                        h = o[0] if isinstance(o, tuple) else o
                        assert h.shape[-1] == add.shape[0], (h.shape, add.shape)
                        if not ratios:  # the prompt pass: ||c v|| / ||h|| per real (non-pad) token
                            ratios.append(h.float().norm(dim=-1))
                        new = steer_hidden(h, add, norm)
                        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new

                    handle = layers[layer].register_forward_hook(hook)
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                torch.manual_seed(a.seed + b)
                try:
                    with torch.no_grad():
                        gen = hf.generate(**enc, do_sample=True, temperature=spec.temperature, top_p=spec.top_p,
                                          top_k=spec.top_k, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
                finally:
                    if handle:
                        handle.remove()
                per_q_ratio = [None] * len(idx)
                if ratios:
                    norms, mask = ratios[0], enc.attention_mask.bool()
                    cv = float(add.norm())
                    per_q_ratio = [float((cv / norms[k][mask[k]]).median()) for k in range(len(idx))]
                for k, i in enumerate(idx):
                    ids = gen[k, enc.input_ids.shape[1]:].tolist()
                    n = next((j + 1 for j, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id)), len(ids))
                    ids = ids[:n]
                    text = tok.decode(ids, skip_special_tokens=True)
                    finish = "stop" if n < a.max_new_tokens or (ids and ids[-1] == tok.eos_token_id) else "length"
                    got, ok = score(text, obs[i], items[i]["answer"])
                    fh.write(json.dumps({"condition": cond, "trait": trait, "coef": coef, "layer": layer,
                                         "question": a.start + i, "answer": items[i]["answer"], "parsed": got,
                                         "correct": int(ok), "n_tokens": n, "finish": finish, "ratio": per_q_ratio[k],
                                         "completion": text}, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"{cond} {trait} {coef:+g}: {len(todo)} questions in {time.time() - t0:.0f}s", flush=True)
    report(out)


if __name__ == "__main__":
    main()
