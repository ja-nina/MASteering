"""The games' own scores next to TextArena's win/lose reward: reading them, logging them, replaying old episodes,
and the report using them for payoffs (and the win/lose reward for who won)."""
import json

from testbed.envs.textarena import scores as S
from testbed.envs.textarena.ta_adapter import TextArenaAdapter


class _State:
    def __init__(self, gs):
        self.game_state = gs


class _Wrapper:  # ta.make wraps the game: the state is on the inner env
    def __init__(self, env):
        self.env = env


def _env(gs, **attrs):
    e = type("E", (), {})()
    e.state = _State(gs)
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


def test_public_goods_tokens_under_wrappers():
    env = _Wrapper(_Wrapper(_env({"total_scores": {0: 31.5, 1: 20, 2: 20, 3: 44}})))
    assert S.game_scores("PublicGoodsGame-v0", env) == {"player_0": 31.5, "player_1": 20.0, "player_2": 20.0,
                                                        "player_3": 44.0}
    assert S.score_name("PublicGoodsGame-v0") == "tokens"


def test_negotiation_value_change_and_auction_attribute():
    neg = _env({"inventory_value": {0: {"initial": 100, "current": 112, "change": 12},
                                    1: {"initial": 90, "current": 85, "change": -5}}})
    assert S.game_scores("SimpleNegotiation-v0", neg) == {"player_0": 12.0, "player_1": -5.0}
    auc = _env({}, auction_results={"player_net_worth": {0: 1200, 1: 950}})
    assert S.game_scores("SimpleBlindAuction-v0", auc) == {"player_0": 1200.0, "player_1": 950.0}


def test_games_without_a_score_or_unreadable():
    assert S.game_scores("SecretMafia-v0", _env({"total_scores": {0: 1}})) is None
    assert S.game_scores("PublicGoodsGame-v0", _env({})) is None  # not there (yet): None, not an error
    assert S.game_scores("PublicGoodsGame-v0", None) is None


def test_adapter_reads_the_score():
    adapter = TextArenaAdapter(env_id="ThreePlayerGOPS-v0", num_players=3,
                               _env=_env({"player_scores": {0: 30, 1: 41, 2: 20}}))
    assert adapter.game_scores() == {"player_0": 30.0, "player_1": 41.0, "player_2": 20.0}


# ---- replaying an episode logged before game_scores

class _Game:
    """A 2-player TextArena-like game: players alternate, '[n]' adds n to the mover's score, 4 moves; the winner
    (+1) is the higher score, the other gets -1. The deal depends on the seed: player 0 starts with seed % 5."""

    def reset(self, num_players, seed=None):
        self.state = _State({"total_scores": {0: float((seed or 0) % 5), 1: 0.0}})
        self.turn = 0

    def get_observation(self):
        return self.turn % 2, "move"

    def step(self, action):
        gs = self.state.game_state["total_scores"]
        gs[self.turn % 2] += float(str(action).strip("[]") or 0) if str(action).strip("[]").isdigit() else 0.0
        self.turn += 1
        return self.turn >= 4, {}

    def close(self):
        gs = self.state.game_state["total_scores"]
        w = 0 if gs[0] > gs[1] else 1
        return {w: 1, 1 - w: -1}, {}


def _logged_episode(tmp_path, moves, seed_ep=0):
    from testbed.config import RunConfig
    run = {"run_id": "public_goods_2p__none", "model": "m", "episodes": 1,
           "game": {"family": "textarena", "id": "PublicGoodsGame-v0", "players": 2}}
    d = tmp_path / "logs" / "public_goods_2p" / run["run_id"]
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps(run))
    # play it once as the run would, to log the JSONL and the summary
    env = TextArenaAdapter(env_id="PublicGoodsGame-v0", num_players=2, _env=_Game())
    env._seed = RunConfig.from_dict(run).episode_seed(seed_ep)
    env.reset()
    with open(d / f"episode_{seed_ep}.jsonl", "w") as f:
        for t, m in enumerate(moves):
            (agent, _), = env.pending()
            step = env.submit({agent: m})
            f.write(json.dumps({"turn": t, "agent_id": agent, "parsed_action": m}) + "\n")
    (d / f"episode_{seed_ep}.summary.json").write_text(json.dumps(
        {"turns": len(moves), "final_rewards": step.rewards, "total_rewards": step.rewards, "agent_stats": {}}))
    return d, env.game_scores()


def _patch_build(monkeypatch):
    import testbed.registry as reg

    def build_game(family, game_id, num_players, env_kwargs):
        a = TextArenaAdapter(env_id=game_id, num_players=num_players, _env=_Game(), **dict(env_kwargs))
        return a, None, None
    monkeypatch.setattr(reg, "build_game", build_game)


def test_replay_reproduces_and_writes_the_score(tmp_path, monkeypatch):
    from scripts import backfill_scores as B
    _patch_build(monkeypatch)
    d, played = _logged_episode(tmp_path, ["[3]", "[9]", "[1]", 0])  # 0: the runner's fallback
    B.main([str(tmp_path / "logs"), "--jobs", "1"])
    side = json.loads((d / "episode_0.scores.json").read_text())
    assert side["game_scores"] == played and side["source"] == "replay" and side["score"] == "tokens"


def test_replay_that_diverges_is_not_kept(tmp_path, monkeypatch):
    from scripts import backfill_scores as B
    _patch_build(monkeypatch)
    d, _ = _logged_episode(tmp_path, ["[3]", "[9]", "[1]", "[2]"])
    s = json.loads((d / "episode_0.summary.json").read_text())
    s["final_rewards"] = {k: -v for k, v in s["final_rewards"].items()}  # the log says the other player won
    (d / "episode_0.summary.json").write_text(json.dumps(s))
    B.main([str(tmp_path / "logs"), "--jobs", "1"])
    assert not (d / "episode_0.scores.json").exists()


# ---- summary -> CSV -> report

def test_summarize_takes_summary_then_replay(tmp_path):
    from scripts.summarize_runs import episode_scores
    sp = tmp_path / "episode_0.summary.json"
    assert episode_scores({"game_scores": {"player_0": 5.0}}, sp) == ({"player_0": 5.0}, "summary")
    (tmp_path / "episode_0.scores.json").write_text(json.dumps({"game_scores": {"player_0": 7.0}}))
    assert episode_scores({}, sp) == ({"player_0": 7.0}, "replay")
    assert episode_scores({"game_scores": None}, tmp_path / "episode_1.summary.json") == ({}, None)


def test_report_uses_scores_never_mixed_with_wins():
    from scripts import report as R
    rows = [{"cell": ("pg", "text"), "reward": 1.0, "score": 40.0},
            {"cell": ("pg", "text"), "reward": -1.0, "score": None},   # not backfilled yet
            {"cell": ("avalon", "text"), "reward": 1.0, "score": None}]
    R.use_scores(rows)
    assert [r["reward"] for r in rows] == [40.0, None, 1.0]
    assert [r["win"] for r in rows] == [1.0, -1.0, 1.0]
    assert R.SCORED_CELLS == {("pg", "text"): (1, 1)}
