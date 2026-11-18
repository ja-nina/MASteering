from pathlib import Path

import pytest

from scripts.make_debate_data import hle_item

from testbed.envs.symbolic.qa_debate import normalize, questions
from testbed.registry import build_game
from testbed.types import ParsedAction, ParseError


def game(n, dataset="gsm8k", episode=0):
    env, renderer, parser = build_game(family="symbolic", game_id="qa_debate", num_players=n,
                                       env_kwargs={"seed": 123, "episode": episode, "dataset": dataset})
    env.reset()
    return env, renderer, parser


def test_hle_item_splits_choices_off_the_question():
    row = {"id": "x1", "category": "Math", "answer": "B",
           "question": "Which is prime?\n\nAnswer Choices:\nA. 4\nB. 7\nC. 9\n  which is odd\nD. 10"}
    it = hle_item(row)
    assert it == {"id": "hle_x1", "question": "Which is prime?", "options": ["4", "7", "9\nwhich is odd", "10"],
                  "answer": "B", "category": "Math"}
    assert hle_item({**row, "answer": "E"}) is None  # answer not among the options
    assert hle_item({**row, "question": "Which is prime? 7"}) is None  # no choices block
    assert hle_item({**row, "question": row["question"].replace("B. 7", "C. 7")}) is None  # letters out of order


def test_question_follows_episode_not_players_or_seed():
    a, _, _ = game(2, episode=7)
    b, _, _ = game(5, episode=7)
    assert a.item["id"] == b.item["id"] == questions("gsm8k")[7]["id"]


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("dataset", ["gsm8k", "mmlu_pro", "supergpqa", "hle"])
def test_full_episode(n, dataset):
    if dataset == "hle" and not (Path(__file__).resolve().parents[1] / "testbed/data/hle.jsonl").exists():
        pytest.skip("testbed/data/hle.jsonl is built locally (not in git): scripts/make_debate_data.py --hle")
    env, renderer, parser = game(n, dataset)
    right = env.correct
    wrong = "B" if right == "A" else "A" if dataset != "gsm8k" else str(int(right) + 1)
    rounds = 0
    while True:
        pend = env.pending()
        assert len(pend) == n
        acts = {}
        for i, (aid, obs) in enumerate(pend):
            prompt = renderer.system_prompt(aid, obs) + renderer.render(obs, aid, env.context)
            assert obs["question"] in prompt
            if rounds:
                assert "THE OTHER AGENTS" in prompt and "my reasoning" in prompt
            ans = right if (i == 0 or rounds == 2) else wrong
            res = parser.parse(f"my reasoning {i}\nANSWER: {ans}", obs, aid, env.context)
            assert isinstance(res, ParsedAction), res
            acts[aid] = res.value
        step = env.submit(acts)
        rounds += 1
        if step.done:
            break
        assert step.info["n_correct"] == 1 and set(step.rewards.values()) == {0.0}
    assert rounds == 3
    assert step.info["n_correct"] == n and step.info["majority_correct"] == 1.0
    assert env.game_scores() == {f"player_{i}": 1.0 for i in range(n)}


def test_fallback_and_tie():
    env, _, _ = game(2)
    step = env.submit({"player_0": {"answer": env.correct, "text": ""}, "player_1": 0})  # runner fallback
    assert step.info["answers"]["player_1"] is None and step.info["majority_correct"] == 1.0
    env, _, _ = game(2)
    other = str(int(env.correct) + 1)
    step = env.submit({"player_0": {"answer": env.correct}, "player_1": {"answer": other}})
    assert step.info["majority"] is None and step.info["majority_correct"] == 0.5


@pytest.mark.parametrize("text,answer", [("so ANSWER: 1,234", "1234"), ("**ANSWER:** $18", "18"),
                                         ("ANSWER: 5 then ANSWER: 7.", "7"), ("ANSWER: -3.50", "-3.50")])
def test_parse_number(text, answer):
    env, _, parser = game(2)
    obs = env.pending()[0][1]
    res = parser.parse(text, obs, "player_0", env.context)
    assert isinstance(res, ParsedAction) and res.value["answer"] == answer
    assert normalize("18.0", "gsm8k") == "18" and normalize("-3.50", "gsm8k") == "-3.5"


def test_parse_letter_and_errors():
    env, _, parser = game(2, "mmlu_pro")
    obs = env.pending()[0][1]
    assert parser.parse("ANSWER: c", obs, "player_0", env.context).value["answer"] == "C"
    assert isinstance(parser.parse("I think C", obs, "player_0", env.context), ParseError)
    env, _, parser = game(2)
    assert isinstance(parser.parse("ANSWER: about ten", env.pending()[0][1], "player_0", env.context), ParseError)
