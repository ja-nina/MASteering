import random

import pytest
import torch

from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.envs.symbolic.werewolf import WerewolfAdapter
from testbed.steering import SteerGroup, SteeringPlan, VectorStore, random_control
from tests.fakes import TRAITS, make_vectors


@pytest.fixture
def store(tmp_path):
    return VectorStore(make_vectors(tmp_path / "vec"), cache_dir=tmp_path / "cache")


def test_store_lists_traits_without_raw_files(store):
    assert store.traits() == sorted(TRAITS)
    assert store.num_layers() == 6
    assert store.candidate_layers() == [1, 2, 3]


def test_best_layer_is_largest_delta_at_max_coefficient(store):
    assert store.best_layer("evil") == 1
    assert store.best_layer("Sarcastic") == 2


def test_single_trait_uses_persvecfast_export(store, tmp_path):
    path = store.steer_file({"evil": 1.0}, 2)
    assert path.endswith("vllm_hook\\evil_L2.pt") or path.endswith("vllm_hook/evil_L2.pt")


def test_layer_without_export_and_mixes_are_written_to_cache(store):
    path = store.steer_file({"evil": 1.0}, 5)
    assert torch.allclose(torch.load(path)["dir"], store.vectors("evil")[5])
    mix = store.steer_file({"evil": 1.0, "sarcastic": 0.5}, 2)
    expected = store.vectors("evil")[2] + 0.5 * store.vectors("sarcastic")[2]
    assert torch.allclose(torch.load(mix)["dir"], expected)


def test_random_control_matches_norm_not_direction(store):
    v = store.vectors("evil")[2]
    r = torch.load(store.steer_file({"evil": 1.0}, 2, random_ctrl=True))["dir"]
    assert torch.isclose(r.norm(), v.norm())
    assert torch.allclose(r, random_control(v, 2))
    assert torch.nn.functional.cosine_similarity(r, v, dim=0) < 0.99


def test_unknown_trait_lists_known_ones(store):
    with pytest.raises(FileNotFoundError, match="known traits"):
        store.vectors("kind")


def test_group_parsing_and_validation():
    g = SteerGroup.from_dict({"trait": "evil", "coefficient": 2.5})
    assert g.traits == {"evil": 1.0} and g.layer == "auto" and g.select == "all"
    assert SteerGroup.from_dict({"traits": ["evil", "empathy"], "coefficient": 1}).traits == {"evil": 1.0, "empathy": 1.0}
    for bad in ({"coefficient": 1}, {"trait": "evil", "coefficient": 1, "select": "some"},
                {"trait": "evil", "coefficient": 1, "select": "fraction"},
                {"trait": "evil", "coefficient": 1, "team": "village"}):
        with pytest.raises(ValueError):
            SteerGroup.from_dict(bad)


def test_resolve_auto_layer_and_label(store):
    steer = SteerGroup.from_dict({"trait": "sarcastic", "coefficient": -1.25}).resolve(store)
    assert steer.layer == 2 and steer.coeff == -1.25
    assert steer.label == "sarcastic@L2x-1.25"
    rnd = SteerGroup.from_dict({"trait": "evil", "coefficient": 2.5, "layer": 3, "random_control": True}).resolve(store)
    assert rnd.label == "random(evil)@L3x2.5" and rnd.vector_path.endswith("evil_L3_random.pt")


def _gbs(n=10):
    env = GBSAdapter(num_players=n)
    env.reset()
    return env


def test_select_all_one_count_fraction(store):
    env = _gbs()
    plan = SteeringPlan.from_config([{"trait": "evil", "coefficient": 1, "select": "one"}], store)
    assert len(plan.assign(env, 0)) == 1
    assert plan.assign(env, 7) == plan.assign(env, 7)  # deterministic per episode seed
    assert len({tuple(plan.assign(env, s)) for s in range(20)}) > 1  # the seat varies across episodes
    plan = SteeringPlan.from_config({"trait": "evil", "coefficient": 1, "select": "fraction", "fraction": 0.3}, store)
    assert len(plan.assign(env, 0)) == 3
    plan = SteeringPlan.from_config({"trait": "evil", "coefficient": 1, "select": "count", "count": 4}, store)
    assert len(plan.assign(env, 0)) == 4
    plan = SteeringPlan.from_config({"trait": "evil", "coefficient": 1}, store)
    assert sorted(plan.assign(env, 0)) == sorted(env.agent_ids())
    assert SteeringPlan.from_config(None, store).assign(env, 0) == {}


def test_pool_by_seat_and_agent_id(store):
    env = _gbs(4)
    plan = SteeringPlan.from_config({"trait": "evil", "coefficient": 1, "agents": [0, "player_2"]}, store)
    assert sorted(plan.assign(env, 0)) == ["player_0", "player_2"]


def test_team_groups_follow_roles_each_episode(store):
    plan = SteeringPlan.from_config([
        {"team": "evil", "trait": "evil", "coefficient": 2.5},
        {"team": "good", "select": "one", "trait": "empathy", "coefficient": 2.5},
    ], store)
    for seed in (1, 2):
        env = WerewolfAdapter(num_players=8, seed=seed)
        env.reset()
        assigned = plan.assign(env, seed)
        teams = env.teams()
        evil = {a for a, s in assigned.items() if s.label.startswith("evil")}
        good = {a for a, s in assigned.items() if s.label.startswith("empathy")}
        assert evil == {a for a, t in teams.items() if t == "evil"}
        assert len(good) == 1 and teams[next(iter(good))] == "good"


def test_team_steering_needs_a_role_game(store):
    plan = SteeringPlan.from_config({"team": "evil", "trait": "evil", "coefficient": 1}, store)
    with pytest.raises(ValueError, match="no teams"):
        plan.assign(_gbs(), 0)


def test_overlapping_groups_are_rejected(store):
    plan = SteeringPlan.from_config([{"trait": "evil", "coefficient": 1},
                                     {"agents": [0], "trait": "empathy", "coefficient": 1}], store)
    with pytest.raises(ValueError, match="two steering groups"):
        plan.assign(_gbs(), 0)
