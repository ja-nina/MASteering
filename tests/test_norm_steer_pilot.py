"""The pure parts of scripts/norm_steer_pilot.py (the GPU part runs on the cluster only)."""
import json

import torch

from scripts.norm_steer_pilot import raw_obs, report, score, steer_hidden
from testbed.envs.symbolic.qa_debate import questions


def test_norm_preserving_keeps_each_tokens_norm_and_moves_towards_v():
    h = torch.randn(2, 5, 8)
    v = torch.randn(8)
    plain, kept = steer_hidden(h, v, False), steer_hidden(h, v, True)
    assert torch.allclose(plain, h + v)
    assert torch.allclose(kept.norm(dim=-1), h.norm(dim=-1), atol=1e-5)
    cos = lambda x: torch.nn.functional.cosine_similarity(x, v.expand_as(x), dim=-1)
    assert (cos(kept) > cos(h)).all()  # same direction change as plain add
    assert torch.allclose(cos(kept), cos(plain), atol=1e-5)


def test_prompt_and_scoring_follow_the_game(tmp_path, capsys):
    from testbed.renderers.symbolic.qa_debate import QADebateRenderer
    item = questions("supergpqa")[0]
    o = raw_obs(item)
    r = QADebateRenderer()
    assert "answering a question over 3 rounds" in r.system_prompt("player_0", o)
    assert "Round 1 of 3." in r.render(o, "player_0", None) and "A. " in r.render(o, "player_0", None)
    assert score(f"so it is\nANSWER: {item['answer']}", o, item["answer"]) == (item["answer"], True)
    assert score("no idea", o, item["answer"]) == (None, False)
    out = tmp_path / "p.jsonl"
    rec = lambda cond, trait, coef, q, ok: {"condition": cond, "trait": trait, "coef": coef, "question": q,
                                            "correct": ok, "n_tokens": 10, "finish": "stop", "parsed": "A",
                                            "ratio": None if cond == "none" else 0.1}
    rows = [rec("none", "-", 0.0, q, 1) for q in range(3)] + [rec("add", "evil", 1.0, q, q % 2) for q in range(3)]
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    report(out)
    text = capsys.readouterr().out
    assert "evil" in text and "add" in text and "-0.667" in text
