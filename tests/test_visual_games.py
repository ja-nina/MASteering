import json
import sys

import pytest
from PIL import Image

from testbed.config import RunConfig, build_jobs
from testbed.envs.meltingpot.adapter import MeltingPotAdapter
from testbed.envs.meltingpot.backend import WorkerSubstrate, action_name
from testbed.envs.symbolic.board import BoardAdapter
from testbed.envs.symbolic.hanabi import HanabiAdapter
from testbed.parsers.meltingpot import MeltingPotParser
from testbed.parsers.symbolic.board import BoardParser
from testbed.registry import build_game
from testbed.renderers.meltingpot import MeltingPotRenderer
from testbed.runner import run_jobs
from testbed.steering import VectorStore
from testbed.types import ParsedAction, ParseError
from testbed.visual import fit, observe
from tests.fakes import FakeGenerator, make_vectors

VISUAL = [("gbs", 3, {}), ("gbs", 3, {"hide_group_size": True}), ("beauty_contest", 3, {}), ("harvest", 3, {}),
          ("werewolf", 6, {}), ("avalon", 6, {}), ("hanabi", 3, {}), ("overcooked", 2, {}),
          ("tictactoe", 2, {}), ("connect_four", 2, {}), ("gomoku", 3, {})]


@pytest.mark.parametrize("game,players,kwargs", VISUAL)
def test_every_mode_renders(game, players, kwargs):
    env, renderer, _ = build_game(family="symbolic", game_id=game, num_players=players, env_kwargs=kwargs)
    env.reset()
    if game in ("gbs", "beauty_contest"):  # one round played, so there is a chart to draw
        env.submit({a: 5 for a, _ in env.pending()})
    aid, obs = env.pending()[0]
    text, none = observe(renderer, "text", obs, aid, env.context)
    visual, images = observe(renderer, "image", obs, aid, env.context)
    both, images2 = observe(renderer, "both", obs, aid, env.context)
    assert none == [] and both == text and len(images) == len(images2) >= 1
    assert all(isinstance(im, Image.Image) and min(im.size) >= 100 for im in images)
    assert "image" in visual and visual != text


def test_image_text_leaves_out_what_the_image_shows():
    env, renderer, _ = build_game(family="symbolic", game_id="hanabi", num_players=3, env_kwargs={})
    env.reset()
    aid, obs = env.pending()[0]
    visual, _ = observe(renderer, "image", obs, aid, env.context)
    assert "'s hand:" not in visual and "Fireworks:" not in visual and "It is YOUR turn" in visual


def test_fit_scales_longer_side():
    assert fit(Image.new("RGB", (88, 44)), 352).size == (352, 176)
    assert fit(Image.new("RGB", (1000, 500)), 500).size == (500, 250)


def test_tictactoe_win_and_turns():
    env = BoardAdapter(variant="tictactoe", seed=3)
    first = env.pending()[0][0]
    for move in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]:  # the first player completes the top row
        aid, _ = env.pending()[0]
        res = env.submit({aid: move})
    assert res.done and res.rewards[first] == 1.0 and res.info["winner"] == first and res.info["cell"] == "C1"


def test_connect_four_gravity_and_draw():
    env = BoardAdapter(variant="connect_four")
    aid, _ = env.pending()[0]
    env.submit({aid: (5, 3)})
    assert env.context.extra["board"][5][3] is not None
    assert (4, 3) in env.legal_actions(aid) and (5, 3) not in env.legal_actions(aid)
    small = BoardAdapter(variant="tictactoe", rows=1, cols=2, k=3)
    for _ in range(2):
        a, _ = small.pending()[0]
        res = small.submit({a: small.legal_actions(a)[0]})
    assert res.done and set(res.rewards.values()) == {0.5}


def test_board_parser():
    parser = BoardParser()
    env = BoardAdapter(variant="tictactoe")
    aid, obs = env.pending()[0]
    assert parser.parse("I take the centre. MOVE: B2", obs, aid, None) == ParsedAction((1, 1))
    assert isinstance(parser.parse("MOVE: D9", obs, aid, None), ParseError)
    c4 = BoardAdapter(variant="connect_four")
    aid, obs = c4.pending()[0]
    assert parser.parse("MOVE: d", obs, aid, None) == ParsedAction((5, 3))


def test_meltingpot_action_names():
    assert action_name({"move": 0, "turn": 0, "fireZap": 0}) == "NOOP"
    assert action_name({"move": 2, "turn": 0, "fireZap": 0}) == "STEP_RIGHT"
    assert action_name({"move": 0, "turn": -1, "fireClean": 0}) == "TURN_LEFT"
    assert action_name({"move": 0, "turn": 0, "fireClean": 1}) == "FIRE_CLEAN"


def test_meltingpot_plans_run_until_someone_needs_a_new_one():
    env = MeltingPotAdapter(num_players=2, substrate="fake:100", max_steps=10)
    env.reset()
    assert [a for a, _ in env.pending()] == ["player_0", "player_1"]
    res = env.submit({"player_0": ["FORWARD", "FORWARD", "FORWARD"], "player_1": ["NOOP"]})
    assert res.info["steps_run"] == 1 and res.rewards == {"player_0": 1.0, "player_1": 0.0}
    assert [a for a, _ in env.pending()] == ["player_1"]  # player_0 still has 2 queued steps
    res = env.submit({"player_1": ["FORWARD", "FORWARD", "FORWARD", "FORWARD"]})
    assert res.info["steps_run"] == 2 and res.info["totals"] == {"player_0": 3.0, "player_1": 2.0}
    while not res.done:
        res = env.submit({a: ["FORWARD"] * 4 for a, _ in env.pending()})
    assert env.context.extra["step"] == 10 and res.rewards == res.info["totals"] and env.pending() == []
    env.close()


def test_meltingpot_renderer_and_parser():
    env = MeltingPotAdapter(num_players=2, substrate="fake:5", global_view=True)
    env.reset()
    aid, obs = env.pending()[0]
    r = MeltingPotRenderer()
    assert r.modes == ("image",) and "FIRE_ZAP" in r.system_prompt(aid, obs)
    text, images = observe(r, "image", obs, aid, env.context)
    assert len(images) == 2 and "ACTIONS:" in text
    parser = MeltingPotParser()
    assert parser.parse("ACTIONS: forward, Turn Left, FIRE_ZAP.", obs, aid, None) == \
        ParsedAction(["FORWARD", "TURN_LEFT", "FIRE_ZAP"])
    assert isinstance(parser.parse("ACTIONS: JUMP", obs, aid, None), ParseError)


def test_meltingpot_worker_process():
    sub = WorkerSubstrate("fake:3", 2, sys.executable)
    assert "FIRE_ZAP" in sub.action_names
    obs = sub.reset()
    assert obs[0]["RGB"].shape == (40, 40, 3)
    _, rewards, done = sub.step([1, 0])
    assert rewards == [1.0, 0.0] and not done
    sub.close()
    assert sub.proc.returncode == 0


def _cfg(tmp_path, run_id, family, game, players, observation, **kw):
    return RunConfig.from_dict({"run_id": run_id, "model": "qwen3.5-9b", "vectors_dir": str(tmp_path / "vec"),
                                "game": {"family": family, "id": game, "players": players,
                                         "env_kwargs": kw.pop("env_kwargs", {})},
                                "episodes": 1, "observation": observation, "logging_dir": str(tmp_path / "logs"),
                                **kw})


def test_image_runs_send_images_and_save_them(tmp_path):
    store = VectorStore(make_vectors(tmp_path / "vec"), cache_dir=tmp_path / "cache")
    configs = [_cfg(tmp_path, "c4_img", "symbolic", "connect_four", 2, "image", save_images=True, image_size=256,
                    steering={"select": "one", "trait": "evil", "coefficient": 2.5}, max_parse_retries=6),
               _cfg(tmp_path, "mp", "meltingpot", "fake:12", 3, "image", env_kwargs={"max_steps": 12}),
               _cfg(tmp_path, "c4_txt", "symbolic", "connect_four", 2, "text", max_parse_retries=6)]
    gen = FakeGenerator()
    assert run_jobs(build_jobs(configs, store), gen) == {"ok": 3, "failed": 0}
    reqs = [r for b in gen.batches for r in b]
    assert all(len(r.images) == 1 and max(r.images[0].size) == 256 for r in reqs if r.key[0] == 0)
    assert all(len(r.images) == 1 for r in reqs if r.key[0] == 1)
    assert all(not r.images for r in reqs if r.key[0] == 2)
    rec = json.loads((configs[0].run_dir / "episode_0.jsonl").read_text().splitlines()[0])
    assert (configs[0].run_dir / rec["images"][0]).exists()
    rec = json.loads((configs[1].run_dir / "episode_0.jsonl").read_text().splitlines()[0])
    assert rec["images"] == 1  # not saved: just the count
    summary = json.loads((configs[1].run_dir / "episode_0.summary.json").read_text())
    assert summary["final_rewards"] == summary["final_info"]["totals"]


def test_text_observation_is_rejected_for_meltingpot(tmp_path):
    store = VectorStore(make_vectors(tmp_path / "vec"), cache_dir=tmp_path / "cache")
    cfg = _cfg(tmp_path, "mp_text", "meltingpot", "fake:5", 2, "text")
    assert run_jobs(build_jobs([cfg], store), FakeGenerator()) == {"ok": 0, "failed": 1}
