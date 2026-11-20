"""Pilot (2026-11-20): steer-to-target, two-sided (vLLM-Hook adjust_rs) vs one-sided (push only towards the trait).

debate_fix's `target` sets every generated token's projection on the trait's unit vector u to a level T:
    two-sided   h += (T - h.u) u                      tokens already past T are pulled back to it
    one-sided   c > 0: h += max(T - h.u, 0) u          a floor: only tokens short of T are pushed up
                c < 0: h += min(T - h.u, 0) u          a ceiling: only tokens past T are pushed down
T and u are exactly debate_fix's vector files (testbed.steering.VectorStore.steer_file(..., target=c): T = the
midpoint of the trait's positive and negative extraction answers + c * |v| / 2), at the same layer, generated
tokens only (phase: decode: the prompt pass is not touched). vLLM-Hook has no one-sided mode, hence HF here.

Conditions per trait:coefficient, on the same questions (a lone agent's round-1 answer, as norm_steer_pilot.py):
    none, two, one, two_random, one_random     (random: a random direction of |v|, its own midpoint level)
Logged per answer, besides accuracy:
    against     share of generated tokens on the far side of T (c > 0: h.u > T). Two-sided pulls these
                back, against the trait; one-sided leaves them alone. ~0 -> the two are the same thing.
    push        mean push (T - h.u, signed towards the trait) per generated token, in units of |v|:
                add_vector at c = 1 pushes 1.0 on every token
    proj        mean h.u per generated token as steered (one-sided: >= T for c > 0)
    read_L<l>   the answer re-read unsteered: mean projection of its tokens on u at layer l, in |v| units
                relative to the midpoint (+0.5 = the positive answers' mean, -0.5 = the negative ones'),
                the monitor's measure of how much the trait shows in the text
For `none` the natural h.u is logged per trait (read off with a hook that changes nothing), so `against`
there says where unsteered tokens sit relative to each target.

    python scripts/target_clamp_pilot.py --vectors-dir <PersVecFast out>/persona/qwen3.5-2b/bf16 \
        --cell sycophantic:1 --cell assertive:1 --n 60 --out ~/clamp_pilot.jsonl
    python scripts/target_clamp_pilot.py ... --out ~/clamp_pilot.jsonl --report-only

Same differences from the games as norm_steer_pilot.py (one round, HF sampling, max_new_tokens 4096): compare
conditions with each other here, not with debate_acc's numbers. Resumes per (condition, trait, coef, question).
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
from scripts.norm_steer_pilot import AGENT, done_keys, raw_obs, score  # noqa: E402

CONDITIONS = ("none", "two", "one", "two_random", "one_random")


def clamp_push(p: torch.Tensor, target: float, coef: float, one_sided: bool) -> torch.Tensor:
    """Per-token push along u that brings projection p to target; one-sided keeps only pushes towards the trait."""
    push = target - p
    if one_sided:
        push = push.clamp(min=0) if coef > 0 else push.clamp(max=0)
    return push


def stats(ps: list, n: int, target, coef: float, one_sided: bool, scale: float) -> dict:
    """against / push / proj over the first n - 1 generated tokens (the first one comes from the prompt pass)."""
    if target is None or n <= 1 or not ps:
        return {}
    p = torch.stack(ps[:n - 1]).float()            # natural h.u at each decode step, before the push
    push = clamp_push(p, target, coef, one_sided)
    sign = 1.0 if coef >= 0 else -1.0
    return {"against": float(((p - target) * sign > 0).float().mean()),
            "push": float((push * sign).mean() / scale),
            "proj": float(((p + push) - (target - coef * scale / 2)).mean() / scale)}  # rel. to midpoint


def report(path: Path):
    rows = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows[(r["trait"], r["coef"], r["condition"])].append(r)
    base = {r["question"]: r for r in rows.get(("-", 0.0, "none"), [])}
    read_keys = sorted({k.split(":")[0] for v in rows.values() for r in v for k in r if k.startswith("read_")})
    head = f"{'trait':14}{'c':>5} {'cond':11}{'qs':>4}{'acc':>7}{'d none':>8}{'+-SE':>7}{'tokens':>7}{'cut':>6}" \
           f"{'against':>9}{'push':>7}{'proj':>7}" + "".join(f"{k:>9}" for k in read_keys)
    print("\n" + head)
    for trait, coef in sorted({k[:2] for k in rows if k[0] != "-"}):
        for cond in CONDITIONS:
            v = base.values() if cond == "none" else rows.get((trait, coef, cond))
            if not v:
                continue
            v = list(v)
            d = [r["correct"] - base[r["question"]]["correct"] for r in v if r["question"] in base]
            m = sum(d) / len(d) if d else float("nan")
            se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5 if len(d) > 1 else float("nan")

            def avg(key, rs=v):
                xs = [r[key] for r in rs if r.get(key) is not None]
                return sum(xs) / len(xs) if xs else float("nan")
            # none: the natural tokens of this trait against this cell's targets
            sub = (lambda k: f"{k}:{trait}:{coef:g}") if cond == "none" else (lambda k: k)
            print(f"{trait:14}{coef:>+5g} {cond:11}{len(v):>4}{sum(r['correct'] for r in v) / len(v):>7.3f}"
                  f"{m:>+8.3f}{se:>7.3f}{avg('n_tokens'):>7.0f}"
                  f"{100 * sum(r['finish'] == 'length' for r in v) / len(v):>5.1f}%"
                  f"{avg(sub('against')):>9.3f}{avg(sub('push')):>7.3f}{avg(sub('proj')):>7.3f}"
                  + "".join(f"{avg(sub(k) if cond == 'none' else k):>9.3f}" for k in read_keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors-dir", required=True)
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--cell", action="append", required=True, help="trait:coefficient (nonzero), e.g. sycophantic:1")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--read-layers", type=int, nargs="+", default=[12, 17])
    ap.add_argument("--n", type=int, default=60)
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
    from testbed.steering import VectorStore

    spec = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(spec.hf_repo)
    tok.padding_side = "left"
    hf = load_hf(spec.hf_repo).eval()
    layers = decoder_layers(hf)
    hidden = hf.config.hidden_size
    types = getattr(getattr(hf.config, "text_config", hf.config), "layer_types", None)
    print(f"{a.model}: {len(layers)} decoder blocks, hidden {hidden}, steer layer {a.layer}"
          f" ({types[a.layer] if types else 'full attention'}), read layers {a.read_layers}", flush=True)
    assert all(0 <= l < len(layers) for l in [a.layer, *a.read_layers])
    store = VectorStore(a.vectors_dir, cache_dir=out.parent / "_clamp_pilot_vectors")

    def target_of(trait, coef, random):
        """(unit u, level T, |v|) from the same file debate_fix's target runs load."""
        f = torch.load(store.steer_file({trait: 1.0}, a.layer, random, None, coef), map_location="cpu")
        u = f["dir"].float()
        assert u.shape == (hidden,) and abs(float(u.norm()) - 1) < 1e-3, (u.shape, float(u.norm()))
        return u, float(f["avg_proj"]), float(store.vectors(trait)[a.layer].float().norm())

    def unit(trait, layer):
        v = store.vectors(trait)[layer].float()
        return v / v.norm(), float(v.norm())

    items = list(questions("supergpqa"))[a.start:a.start + a.n]
    renderer = QADebateRenderer()
    obs = [raw_obs(it) for it in items]
    prompts = [template(tok, spec, renderer.system_prompt(AGENT, o), renderer.render(o, AGENT, None), 0) for o in obs]

    cells = []
    for cell in a.cell:
        trait, coef = cell.split(":")
        coef = float(coef)
        if coef == 0:
            raise SystemExit("one-sided needs a direction: coefficient must be nonzero")
        cells.append((trait, coef))
        u, T, s = target_of(trait, coef, False)
        mid = T - coef * s / 2
        print(f"{trait} c={coef:+g}: |v| = {s:.2f}, midpoint proj {mid:.2f}, target {T:.2f} "
              f"(positive answers {mid + s / 2:.2f}, negative {mid - s / 2:.2f})", flush=True)
    traits = sorted({t for t, _ in cells})
    # midpoint level on each trait's own unit vector at each read layer, to express read-outs in |v| units
    read_ref = {}
    for t in traits:
        for l in a.read_layers:
            u_l, s_l = unit(t, l)
            read_ref[(t, l)] = (u_l, float(store.midpoint(t, l) @ u_l), s_l)

    jobs = [("none", "-", 0.0, False, False)]
    for trait, coef in cells:
        jobs += [("two", trait, coef, False, False), ("one", trait, coef, True, False),
                 ("two_random", trait, coef, False, True), ("one_random", trait, coef, True, True)]

    done = done_keys(out)
    with open(out, "a", encoding="utf-8") as fh:
        for cond, trait, coef, one_sided, random in jobs:
            todo = [i for i in range(len(items)) if (cond, trait, coef, a.start + i) not in done]
            if not todo:
                continue
            # what the hook reads: the steered (u, T) or, for none, every cell's (u, T) untouched
            if cond == "none":
                watch = {(t, c): target_of(t, c, False) for t, c in cells}
            else:
                watch = {(trait, coef): target_of(trait, coef, random)}
            t0 = time.time()
            for b in range(0, len(todo), a.batch):
                idx = todo[b:b + a.batch]
                seen = {k: [] for k in watch}       # per decode step: natural h.u, shape [batch]
                units = {k: (u.cuda(), T) for k, (u, T, _) in watch.items()}

                def hook(m, i, o, steer=cond != "none"):
                    h = o[0] if isinstance(o, tuple) else o
                    if h.shape[1] != 1:              # the prompt pass: phase decode leaves it alone
                        return None
                    assert h.shape[-1] == hidden, h.shape
                    for k, (u, T) in units.items():
                        p = (h[:, 0].float() @ u)
                        seen[k].append(p.cpu())
                    if not steer:
                        return None
                    (u, T), = units.values()
                    push = clamp_push(seen[next(iter(units))][-1].cuda(), T, coef, one_sided)
                    new = h + (push[:, None, None] * u).to(h.dtype)
                    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new

                handle = layers[a.layer].register_forward_hook(hook)
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                torch.manual_seed(a.seed + b)
                try:
                    with torch.no_grad():
                        gen = hf.generate(**enc, do_sample=True, temperature=spec.temperature, top_p=spec.top_p,
                                          top_k=spec.top_k, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
                finally:
                    handle.remove()
                for k, i in enumerate(idx):
                    ids = gen[k, enc.input_ids.shape[1]:].tolist()
                    n = next((j + 1 for j, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id)), len(ids))
                    ids = ids[:n]
                    text = tok.decode(ids, skip_special_tokens=True)
                    finish = "stop" if n < a.max_new_tokens or (ids and ids[-1] == tok.eos_token_id) else "length"
                    got, ok = score(text, obs[i], items[i]["answer"])
                    rec = {"condition": cond, "trait": trait, "coef": coef, "layer": a.layer,
                           "question": a.start + i, "answer": items[i]["answer"], "parsed": got,
                           "correct": int(ok), "n_tokens": n, "finish": finish}
                    for (t, c), (_, T, s) in watch.items():
                        st = stats([p[k] for p in seen[(t, c)]], n, T, c, one_sided, s)
                        if cond == "none":
                            rec.update({f"{key}:{t}:{c:g}": val for key, val in st.items()})
                        else:
                            rec.update(st)
                    # unsteered re-read of prompt + answer: the trait's projection over the answer tokens
                    full = torch.tensor([enc.input_ids[k][enc.attention_mask[k].bool()].tolist() + ids], device="cuda")
                    with torch.no_grad():
                        hs = hf(full, output_hidden_states=True).hidden_states
                    start = full.shape[1] - n
                    for t in (traits if cond == "none" else [trait]):
                        for l in a.read_layers:
                            u_l, mid_l, s_l = read_ref[(t, l)]
                            r = float(((hs[l + 1][0, start:].float() @ u_l.cuda()).mean() - mid_l) / s_l)
                            for c in ([c for tt, c in cells if tt == t] if cond == "none" else [None]):
                                rec[f"read_L{l}" + (f":{t}:{c:g}" if c is not None else "")] = r
                    rec["completion"] = text
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"{cond} {trait} {coef:+g}: {len(todo)} questions in {time.time() - t0:.0f}s", flush=True)
    report(out)


if __name__ == "__main__":
    main()
