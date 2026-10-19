"""Token counts rebuilt from old JSONLs: every attempt's prompt and answer, images by the engine's formula."""
import json

from scripts.backfill_tokens import count_tokens


def test_counts_every_attempt_and_images(tmp_path):
    words = lambda text: len(text.split())  # a tokenizer: one token per word
    jsonl = tmp_path / "episode_0.jsonl"
    recs = [
        {"agent_id": "p0", "model_input": "SYS a b USER one two", "user_prompt": "USER one two", "images": 1,
         "attempts": [{"user_prompt": "USER one", "n_tokens": 5}, {"user_prompt": "USER one two", "n_tokens": 7}]},
        {"agent_id": "p1", "system_prompt": "SYS", "user_prompt": "hi", "n_tokens": 3, "images": 0},
    ]
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in recs))
    got = count_tokens(jsonl, words, per_image=10)
    # p0: around the user prompt ("SYS a b") = 6 - 3 = 3 tokens; attempts read 3+2+10 and 3+3+10; generated 5 + 7
    assert got["p0"] == {"in": 15 + 16, "out": 12}
    assert got["p1"] == {"in": 1 + 1, "out": 3}  # no model_input: system + user
    assert count_tokens(jsonl, words, per_image=None)["p0"]["in"] is None  # image size unknown: not counted
