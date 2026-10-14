import csv
import random

from scripts.report import (baselines, build, detectability, effect_table, is_vector, load, manipulation,
                            obs_compare, paired, readout_slope, reward_scale, zscored)

COEFS = [-2, -1, 1, 2]


def _rows(rng):
    """Two games (game A also as images), 3 agents, 40 seeds. The steered agent's own-trait readout rises 0.5 per unit coefficient
    (vector only); in game A, an evil-steered agent's reward rises 0.3 per unit; the random direction does nothing."""
    out = []

    def add(run_id, cond, ep, agent, steered, trait=None, coef=None, via=None, rnd=False, reward=0.0, mon=None):
        out.append({"run_id": run_id, "group": run_id.split("__")[0], "condition": cond, "game": "g",
                    "players": 3, "observation": obs, "episode": ep, "agent": agent,
                    "condition_select": "one" if trait else "", "condition_team": "", "condition_trait": trait or "",
                    "condition_coefficient": "" if coef is None else coef, "condition_via": via or "",
                    "condition_random": rnd, "steered": steered, "steer_label": "", "role": "", "team": "",
                    "reward": reward, "turns": 5, "parse_fallbacks": 0, "episode_turns": 5, "truncated": False,
                    "answer_chars": 200 + 20 * abs(coef or 0) if steered else 200,
                    **{f"mon_{t}": v for t, v in (mon or {}).items()}})

    obs = "text"
    for game, obs in (("ga_3p", "text"), ("gb_3p", "text"), ("ga_3p", "image")):
        sfx = "" if obs == "text" else f"__obs-{obs}"
        for ep in range(40):
            seat_reward = {f"p{i}": rng.gauss(0, 1) for i in range(3)}
            base_mon = lambda: {"evil": rng.gauss(0, 0.2), "sarcastic": rng.gauss(0, 0.2)}  # noqa: E731
            for a in seat_reward:
                add(f"{game}__none" + sfx, "none", ep, a, False, reward=seat_reward[a], mon=base_mon())
            for trait in ("evil", "sarcastic"):
                for c in COEFS:
                    for cond, via, rnd in (("one", "vector", False), ("all_random", "vector", True)):
                        rid = f"{game}__{cond}__{trait}_x{c}" + sfx
                        for a in seat_reward:
                            st = a == "p0"
                            m = base_mon()
                            if st and not rnd:
                                m[trait] += 0.5 * c
                            elif not rnd:  # contagion: the co-players pick up a little of the trait
                                m[trait] += 0.1 * c
                            bump = 0.3 * c if (st and not rnd and trait == "evil" and game == "ga_3p") else 0.0
                            add(rid, cond, ep, a, st, trait, c, via, rnd, seat_reward[a] + bump + rng.gauss(0, .05), m)
                for c in (-1, 1):
                    rid = f"{game}__prompt_one__{trait}_x{c}" + sfx
                    for a in seat_reward:
                        m = base_mon()
                        if a == "p0":
                            m[trait] += 0.8 * c
                        add(rid, "prompt_one", ep, a, a == "p0", trait, c, "prompt", False, seat_reward[a], m)
    return out


def test_report_recovers_known_effects(tmp_path):
    rows = _rows(random.Random(0))
    path = tmp_path / "agents.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    data = load(path)
    man = manipulation(data, ["evil", "sarcastic"])
    assert abs(man["evil"]["vector"]["b"] - 0.5) < 0.05 and man["evil"]["vector"]["p"] < 1e-6
    assert abs(man["evil"]["random"]["b"]) < 0.05
    assert man["sarcastic"]["prompt"][1][0] > 0.6 and man["sarcastic"]["prompt"][-1][0] < -0.6

    steered = paired(data, baselines(data), lambda r: r["steered"] and r["via"] == "vector" and not r["random"])
    eff = effect_table(steered, ["evil", "sarcastic"], reward_scale(data))
    assert abs(eff["evil"][("ga_3p", "text")]["b"] - 0.3) < 0.02
    assert abs(eff["evil"][("gb_3p", "text")]["b"]) < 0.02
    assert abs(eff["sarcastic"][("ga_3p", "text")]["b"]) < 0.02

    z = zscored(data)  # readouts in SD units of the unsteered agents (SD 0.2 here)
    assert abs(readout_slope(z, "evil", lambda r: is_vector(r))["b"] - 2.5) < 0.2
    contagion = readout_slope(z, "evil", lambda r: r["condition"] == "one" and not r["steered"])
    assert abs(contagion["b"] - 0.5) < 0.1 and contagion["p"] < 1e-6
    off = readout_slope(z, "evil", lambda r: is_vector(r), readout="sarcastic")
    assert abs(off["b"]) < 0.15  # steering evil does not move sarcasm here
    det = detectability(z, "evil")
    assert det[2][0] > 0.95 and det[-2][0] > 0.95 and 0.5 < det[1][0]
    res, both = obs_compare(z, "evil", lambda sub, t: readout_slope(sub, t, lambda r: is_vector(r)))
    assert both == ["ga_3p"] and abs(res["diff"]["b"]) < 0.3  # same effect in text and image here

    page = build(data)
    assert "<svg" in page and "Steering report" in page and "ga_3p" in page
    for section in ("Specificity", "Detectability", "Amplify vs suppress", "Persona prompt vs persona vector", "Per game",
                    "Who is steered", "Contagion", "Text vs image", "Coherence cost"):
        assert section in page, section
    (tmp_path / "report.html").write_text(page, encoding="utf-8")


def test_link_splits_the_effect():
    """Outcome driven by the language: reward = 0.4 x readout, readout = 0.5 x coefficient + noise.
    The coefficient's effect then runs (almost) entirely through the readout."""
    from scripts.report import link, ols
    rng = random.Random(1)
    rows = []
    for ep in range(400):
        c = rng.choice(COEFS)
        z = 0.5 * c + rng.gauss(0, 1)
        rows.append({"trait": "evil", "coef": c, "mon": {"evil": z}, "diff": 0.4 * z + rng.gauss(0, 0.1),
                     "cell": ("g_3p", "text"), "cluster": ("r", ep)})
    res = link(rows, "evil", {})
    assert abs(res["total"]["b"] - 0.2) < 0.03 and abs(res["via_readout"]["b"] - 0.4) < 0.03
    assert abs(res["direct"]["b"]) < 0.03 and res["share"] > 0.85
    xs = [(x, w + (x % 3)) for x, w in zip(range(50), range(50, 0, -1))]
    b = ols([2 * x + 3 * w for x, w in xs], [list(v) for v in xs], list(range(50)))
    assert abs(b[0]["b"] - 2) < 1e-6 and abs(b[1]["b"] - 3) < 1e-6


def test_examples_show_matched_first_answers(tmp_path):
    import json
    from scripts.report import examples_section
    g = tmp_path / "negotiation_2p"
    texts = {"none": "I have analyzed my resources and will offer a fair trade of wheat for brick. " * 3,
             "all__sarcastic_x-2": "A careful, sincere offer. " * 6, "all__sarcastic_x2": "Oh, how WONDERFUL. " * 8,
             "all_random__sarcastic_x2": "Let me analyze the resources first. " * 5,
             "prompt_all__sarcastic_x1": "Oh, fantastic, another trade. " * 5, "prompt_all__sarcastic_x-1": "Sure. " * 30}
    for cond, text in texts.items():
        d = g / f"negotiation_2p__{cond}"
        d.mkdir(parents=True)
        (d / "episode_0.jsonl").write_text(json.dumps({"agent_id": "player_0", "completion": text,
                                                       "monitor": {"16": {"sarcastic": len(cond) / 10}}}) + "\n")
        (d / "episode_0.summary.json").write_text("{}")
    page = examples_section(tmp_path, ["sarcastic"], [-2, -1, 1, 2])
    assert "sarcastic — negotiation_2p, episode 0" in page
    for label in ("unsteered", "vector c=-2", "vector c=2", "random c=2", "prompt +", "prompt −"):
        assert f"<b>{label}</b>" in page
    assert "how WONDERFUL" in page and "sarcastic readout: 1.80" in page


def test_prompt_equivalent_coefficient():
    """Which vector strength a prompt corresponds to, read off the vector's dose-response curve."""
    from scripts.report import cosine, equivalent_coef
    curve = {0.0: 0.0, 0.5: 1.0, 1.0: 2.0, 2.0: 4.0, -1.0: -1.0, -2.0: -2.0}
    assert equivalent_coef(curve, 3.0, 1) == (1.5, "")
    assert equivalent_coef(curve, 9.0, 1) == (2.0, ">")  # stronger than every tested c
    assert equivalent_coef(curve, 0.5, -1) == (0.0, "<")  # the prompt moves the other way
    assert equivalent_coef(curve, -1.5, -1) == (-1.5, "")
    assert abs(cosine({"a": 1, "b": 2}, {"a": 2, "b": 4}) - 1) < 1e-9  # same profile, any size
    assert abs(cosine({"a": 1, "b": 0}, {"a": 0, "b": 3})) < 1e-9  # unrelated profiles
