"""scripts/backfill_monitor.py without vLLM: turns rebuilt from the JSONL as the capture engine reads them, the side
file, and summarize_runs taking the monitor from it."""
import json

import torch
from PIL import Image

from scripts.backfill_monitor import compare, read_episode, side_path, turn_result, write_side
from scripts.summarize_runs import episode_monitor

encode = lambda s: [len(w) for w in s.split()]  # noqa: E731 - one token per word
STOP = 99


def _rec(turn, agent, completion, n_tokens, images=0, finish="stop", monitor=None):
    return {"turn": turn, "agent_id": agent, "model_input": "<sys> play <assistant> ", "completion": completion,
            "n_tokens": n_tokens, "finish_reason": finish, "images": images, "parse_ok": True,
            "parsed_action": "x", "monitor": monitor or {}, "attempts": [{"n_prompt_tokens": 40}]}


def test_text_turn_is_the_prompt_ids_and_the_answer_ids_with_the_stop_token(tmp_path):
    res, bad = turn_result(_rec(0, "a", "go left", 3), tmp_path, encode, STOP)
    assert res.prompt_token_ids == encode("<sys> play <assistant> ") and res.token_ids == encode("go left") + [STOP]
    assert not bad
    res, bad = turn_result(_rec(0, "a", "go left", 2, finish="length"), tmp_path, encode, STOP)
    assert res.token_ids == encode("go left") and not bad  # cut off: no stop token
    assert turn_result(_rec(0, "a", "go left", 5), tmp_path, encode, STOP)[1]  # re-encodes to another length


def test_image_turns_need_their_saved_images(tmp_path):
    assert turn_result(_rec(0, "a", "go", 2, images=1), tmp_path, encode, STOP)[0] is None  # only a count
    (tmp_path / "ep").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "ep" / "t0.png")
    res, bad = turn_result(_rec(0, "a", "go left now", 4, images=["ep/t0.png"]), tmp_path, encode, STOP)
    assert res.prompt_token_ids == [] and len(res.token_ids) == 3 and res.images[0].size == (8, 8)
    assert res.prompt == "<sys> play <assistant> " and res.n_prompt_tokens == 40 and not bad
    res, _ = turn_result(_rec(0, "a", "", 0), tmp_path, encode, STOP)
    assert res.token_ids == []  # nothing generated: no readout, as live


class _Reader:
    def read(self, results):
        return [{3: torch.full((len(r.token_ids), 2), float(len(r.token_ids)))} for r in results]


class _Monitor:
    def scores(self, rows):
        return {} if rows[3].shape[0] == 0 else {"3": {"evil": float(rows[3].mean())}}


def test_episode_to_side_file_to_summary_rows(tmp_path):
    recs = [_rec(0, "a", "go left", 3), _rec(0, "b", "stay", 2), _rec(1, "a", "x y z w", 5), _rec(1, "b", "", 0)]
    spath = tmp_path / "episode_0.summary.json"
    (tmp_path / "episode_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    got_recs, scores, mismatched = read_episode(spath, _Reader(), _Monitor(), encode, STOP)
    assert [s.get("3", {}).get("evil") for s in scores] == [3.0, 2.0, 5.0, None] and mismatched == 0
    write_side(spath, got_recs, scores, [3], mismatched)
    side = json.loads(side_path(spath).read_text(encoding="utf-8"))
    assert side["turns"] == scores and side["agent_stats"]["a"]["monitor"] == {"3": {"evil": 4.0}}
    stats = {"a": {"turns": 2, "monitor": {}}, "b": {"turns": 2, "monitor": {}}}
    filled = episode_monitor(stats, spath)
    assert filled["a"]["monitor"] == {"3": {"evil": 4.0}} and filled["a"]["turns"] == 2
    assert filled["b"]["monitor"] == {"3": {"evil": 2.0}}
    live = {"a": {"turns": 2, "monitor": {"3": {"evil": 1.0}}}}
    assert episode_monitor(live, spath) == live  # the summary's own monitor wins


def test_compare_against_live_scores():
    recs = [_rec(0, "a", "go", 2, monitor={"3": {"evil": 1.0}}), _rec(0, "b", "go", 2, monitor={"3": {"evil": 3.0}})]
    c = compare(recs, [{"3": {"evil": 1.5}}, {"3": {"evil": 3.0}}])
    assert c["n"] == 2 and c["max"] == 0.5 and c["mean"] == 0.25 and c["live_sd"] == 1.0
