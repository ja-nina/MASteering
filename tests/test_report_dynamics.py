"""Over-the-episode, actions and dose-response sections recover effects planted in synthetic games, end to end:
EpisodeLogger -> summary agent_stats -> summarize_runs CSV -> report."""
import json
import random

from scripts import report, summarize_runs
from testbed.logging_.episode_logger import EpisodeLogger

COEFS = (-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2)


def play(logs, coef, ep, rng):
    group = "g_2p"
    run_id = f"{group}__none" if coef == 0 else f"{group}__one__evil_x{coef}"
    d = logs / group / run_id
    d.mkdir(parents=True, exist_ok=True)
    steering = None if coef == 0 else {"select": "one", "trait": "evil", "coefficient": coef}
    (d / "run.json").write_text(json.dumps({"run_id": run_id, "game": {"id": "g", "players": 2}, "steering": steering}))
    log = EpisodeLogger(str(logs / group), run_id, ep)
    for turn in range(6):
        for aid in ("p0", "p1"):
            c = coef if aid == "p0" else 0
            fade = 1 - turn / 6  # the persona fades over the episode
            evil = rng.gauss(0, 1) + 3 * c * fade
            bid = 50 + 10 * c + rng.gauss(0, 5)
            vote = "APPROVE" if rng.random() < 0.5 + 0.2 * c / 2 else "REJECT"
            log.log_step(game="g", turn=turn, agent_id=aid, system_prompt="", user_prompt="", completion="x",
                         parsed_action=f"[Bid {bid:.0f}]" if turn % 2 else vote, parse_retries=0, reward=0.0,
                         monitor={"16": {"evil": evil, "kind": rng.gauss(0, 1)}})
    reward = coef - 0.6 * coef ** 2 + rng.gauss(0, 0.3)  # peaks at c ~ +0.8
    log.close(summary={"final_rewards": {"p0": reward, "p1": 0.0},
                       "steered": {"p0": {"label": f"evil x{coef}"}} if coef else {}})


def test_dynamics_actions_dose_response(tmp_path):
    rng = random.Random(0)
    logs = tmp_path / "main"
    for coef in (0,) + COEFS:
        for ep in range(25):
            play(logs, coef, ep, rng)
    out = tmp_path / "agents.csv.gz"
    summarize_runs.main([str(logs), "--out", str(out), "--jobs", "2"])
    page = report.build(report.load(out))
    s14 = page[page.index("id=s14"):page.index("id=s15")]
    s15 = page[page.index("id=s15"):page.index("id=s16")]
    s16 = page[page.index("id=s16"):page.index("id=games")]
    assert "<b>fades</b> over the episode for: evil" in s14
    assert "mean numeric choice" in s15 and "evil → mean numeric choice +" in s15
    assert "share &#x27;approve&#x27;" in s15 or "share 'approve'" in s15
    assert "∩ peaks at c ≈ +0." in s16
