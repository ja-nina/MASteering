import json

import pytest

from testbed.config import RunConfig, build_jobs
from testbed.runner import run_jobs
from testbed.steering import SteerGroup, VectorStore
from tests.fakes import FakeGenerator, make_vectors


@pytest.fixture
def store(tmp_path):
    return VectorStore(make_vectors(tmp_path / "vec"), cache_dir=tmp_path / "cache", traits_dir=tmp_path / "traits")


def test_default_traits_dir_follows_persvecfast_layout(tmp_path):
    store = VectorStore(tmp_path / "PersVecFast" / "out" / "persona" / "qwen3.5-9b" / "bf16", cache_dir=tmp_path)
    assert store.traits_dir == tmp_path / "PersVecFast" / "data" / "persona_vectors"


def test_prompt_group_has_persona_and_no_vector(store):
    steer = SteerGroup.from_dict({"trait": "sarcastic", "via": "prompt"}).resolve(store)
    assert steer.persona == "You clearly exhibit the trait 'sarcastic'."
    assert steer.vector_path is None and steer.layer is None and steer.label == "prompt(sarcastic)"
    neg = SteerGroup.from_dict({"trait": "sarcastic", "via": "prompt", "coefficient": -1}).resolve(store)
    assert neg.persona == "You do NOT exhibit the trait 'sarcastic'." and neg.label == "prompt(-sarcastic)"


def test_both_group_has_persona_and_vector(store):
    steer = SteerGroup.from_dict({"trait": "evil", "via": "both", "coefficient": 2.5}).resolve(store)
    assert steer.persona and steer.vector_path and steer.layer == 1
    assert steer.label == "prompt(evil)+evil@L1x2.5"
    mix = SteerGroup.from_dict({"traits": {"evil": 1, "empathy": -1}, "via": "prompt"}).resolve(store)
    assert mix.persona.splitlines() == ["You clearly exhibit the trait 'evil'.", "You do NOT exhibit the trait 'empathy'."]


def test_prompt_group_validation():
    for bad in ({"trait": "evil", "via": "telepathy", "coefficient": 1},
                {"trait": "evil", "via": "prompt", "random_control": True},
                {"trait": "evil"}):  # vector groups need a coefficient
        with pytest.raises(ValueError):
            SteerGroup.from_dict(bad)


def test_persona_goes_before_the_system_prompt_of_steered_agents_only(tmp_path, store):
    cfg = RunConfig.from_dict({"run_id": "gbs_prompt", "model": "qwen3.5-9b", "vectors_dir": str(tmp_path / "vec"),
                               "traits_dir": str(tmp_path / "traits"), "episodes": 1,
                               "game": {"family": "symbolic", "id": "gbs", "players": 3, "env_kwargs": {"num_rounds": 2}},
                               "steering": {"agents": [1], "trait": "sarcastic", "via": "prompt"},
                               "logging_dir": str(tmp_path / "logs")})
    gen = FakeGenerator()
    assert run_jobs(build_jobs([cfg], store), gen) == {"ok": 1, "failed": 0}
    for r in (r for b in gen.batches for r in b):
        steered = r.key[3] == "player_1"
        assert r.system.startswith("You clearly exhibit the trait 'sarcastic'.\n\nYou are player_1") == steered
        assert (r.steer is not None) == steered and (r.steer is None or r.steer.vector_path is None)
    rec = next(json.loads(l) for l in (cfg.run_dir / "episode_0.jsonl").read_text().splitlines()
               if json.loads(l)["agent_id"] == "player_1")
    assert rec["steering"]["persona"].startswith("You clearly") and rec["system_prompt"].startswith("You clearly")
