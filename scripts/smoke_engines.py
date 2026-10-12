"""GPU checks for the game engines (run via scripts/slurm/smoke.sbatch).

1. both engines load at once (steer on GPU 0, capture on GPU 1)
2. per-request steering: in ONE batch, steered requests move the monitor's trait projection
   up (positive coefficient) or down (negative), unsteered ones stay at baseline
3. coefficient 0 and no steering give the same text for the same seed
4. capture readout returns one row per completion token (checked inside CaptureEngine)
5. image prompts (Qwen3.5 is a VLM) generate, are read back, and still respond to steering
6. the prompt persona (the trait's PersVecFast instruction in the system prompt) raises the projection
Every completion, its projections and the check results are saved to --out (samples.json, board.png).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.config import DEFAULT_PVF_OUT  # noqa: E402
from testbed.engine import Monitor, TurnRequest  # noqa: E402
from testbed.mia_engine import open_engines  # noqa: E402
from testbed.steering import SteerGroup, VectorStore  # noqa: E402

QUESTIONS = ["What do you think about your neighbour's new car?", "How was your day at work?",
             "Can you recommend a good book?", "What should we do this weekend?",
             "Tell me about the weather today.", "What is your opinion of modern art?"]
FAILED = []
CHECKS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        FAILED.append(name)


def _samples(results, requests, score, labels):
    by_key = {r.key: r for r in requests}
    return [{"condition": r.key[0], "steering": labels.get(r.key[0]), "system": by_key[r.key].system,
             "user": by_key[r.key].user, "images": len(by_key[r.key].images), "completion": r.text,
             "finish_reason": r.finish_reason, "n_tokens": r.n_tokens, "projection": score[r.key]}
            for r in results]


def _board_image():
    from testbed.envs.symbolic.board import BoardAdapter
    from testbed.renderers.symbolic.board import BoardRenderer
    env = BoardAdapter(variant="connect_four")
    for col in (3, 3, 2, 4):
        aid, _ = env.pending()[0]
        env.submit({aid: next(m for m in env.legal_actions(aid) if m[1] == col)})
    aid, obs = env.pending()[0]
    return BoardRenderer().render_images(obs, aid, env.context)[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.5-9b")
    p.add_argument("--trait", default="sarcastic")
    p.add_argument("--coefficient", type=float, default=2.5)
    p.add_argument("--out", default="results/smoke/engines", help="where samples.json and board.png go")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = VectorStore(f"{os.environ.get('PVF_OUT', DEFAULT_PVF_OUT)}/persona/{args.model}/bf16",
                        cache_dir=Path("out/_vectors") / args.model)
    up = SteerGroup.from_dict({"trait": args.trait, "coefficient": args.coefficient}).resolve(store)
    down = SteerGroup.from_dict({"trait": args.trait, "coefficient": -args.coefficient, "layer": up.layer}).resolve(store)
    zero = SteerGroup.from_dict({"trait": args.trait, "coefficient": 0.0, "layer": up.layer}).resolve(store)
    prompt = SteerGroup.from_dict({"trait": args.trait, "via": "prompt"}).resolve(store)
    layers = store.candidate_layers()
    monitor = Monitor(store, layers, [args.trait])
    engine_cfg = {"max_model_len": 8192, "max_images": 1, "steer": {"gpus": "0"}, "monitor": {"gpus": "1"}}

    with open_engines(args.model, engine_cfg, layers) as (gen, reader):
        check("steer + capture engines load side by side", reader is not None)
        conditions = {"none": None, "up": up, "down": down, "zero": zero, "prompt": prompt}
        base = "You are a helpful assistant."
        requests = [TurnRequest(key=(c, i), system=f"{s.persona}\n\n{base}" if s and s.persona else base,
                                user=q, seed=i, steer=s, max_tokens=160)
                    for c, s in conditions.items() for i, q in enumerate(QUESTIONS)]
        results = gen.generate(requests)  # one batch: steered and unsteered requests mixed
        by = {r.key: r for r in results}
        rows = reader.read(results)
        score = {r.key: monitor.scores(x)[str(up.layer)][args.trait] for r, x in zip(results, rows)}
        mean = {c: statistics.mean(score[(c, i)] for i in range(len(QUESTIONS))) for c in conditions}
        print(f"mean {args.trait} projection at layer {up.layer}: {mean}", flush=True)
        check(f"+{args.coefficient} steering raises the {args.trait} projection", mean["up"] > mean["none"])
        check(f"-{args.coefficient} steering lowers the {args.trait} projection", mean["down"] < mean["none"])
        same = sum(by[("zero", i)].text == by[("none", i)].text for i in range(len(QUESTIONS)))
        check("coefficient 0 == unsteered (same seed)", same >= len(QUESTIONS) - 1, f"{same}/{len(QUESTIONS)} identical")
        check(f"the {args.trait} prompt persona raises the {args.trait} projection", mean["prompt"] > mean["none"])
        check("readout has one row per completion token",
              all(x[layers[0]].shape[0] == r.n_tokens for r, x in zip(results, rows)))
        labels = {c: (s.label if s else "none") for c, s in conditions.items()}
        samples = {"model": args.model, "trait": args.trait, "layer": up.layer, "mean_projection": mean,
                   "text": _samples(results, requests, score, labels)}
        for c in ("none", "up", "down", "prompt"):
            print(f"\n--- {c} ({conditions[c].label if conditions[c] else 'unsteered'}):\n{by[(c, 0)].text}", flush=True)

        # image observations: a rendered connect-four position in the prompt
        board = _board_image()
        img_requests = [TurnRequest(key=(c, i), system="You are playing connect four.", seed=i, steer=s,
                                    user="Describe the position in the image, then say which column you would play.",
                                    images=[board], max_tokens=160)
                        for c, s in (("none", None), ("up", up)) for i in range(4)]
        img_results = gen.generate(img_requests)
        check("image prompts generate", all(r.text.strip() for r in img_results))
        img_rows = reader.read(img_results)
        check("image turns are read back (completion rows ~ tokens)",
              all(abs(x[layers[0]].shape[0] - r.n_tokens) <= 2 for r, x in zip(img_results, img_rows)))
        img_score = {r.key: monitor.scores(x)[str(up.layer)][args.trait] for r, x in zip(img_results, img_rows)}
        img_mean = {c: statistics.mean(img_score[(c, i)] for i in range(4)) for c in ("none", "up")}
        print(f"image prompts, mean {args.trait} projection: {img_mean}", flush=True)
        check(f"steering raises the {args.trait} projection with an image in context", img_mean["up"] > img_mean["none"])
        print(f"\n--- image, unsteered:\n{img_results[0].text}\n--- image, steered:\n{img_results[4].text}", flush=True)
        board.save(out / "board.png")
        samples.update(image_mean_projection=img_mean,
                       image=_samples(img_results, img_requests, img_score, {"none": "none", "up": up.label}))
    samples["checks"] = CHECKS
    (out / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"samples -> {out / 'samples.json'}", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
