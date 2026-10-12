import yaml

from scripts.make_configs import expand
from testbed.config import RunConfig

SWEEP = yaml.safe_load("""
name: t
model: qwen3.5-9b
episodes: 3
logging_dir: logs
traits: [evil, empathy]
coefficients: [1.25, 2.5]
conditions:
  none: null
  all: {select: all}
  evil_team: {team: evil}
games:
  gbs: {family: symbolic, id: gbs, players: [3, 6]}
  werewolf: {family: symbolic, id: werewolf, players: 8, roles: true}
""")


def test_expand_cells_and_role_conditions():
    runs = expand(SWEEP)
    # gbs: 2 player counts x (1 baseline + 4 trait/coef cells); werewolf: 1 + 4 + 4
    assert len(runs) == 2 * 5 + 9
    cfg = runs["werewolf_8p/werewolf_8p__evil_team__evil_x2.5.yaml"]
    assert cfg["steering"] == {"team": "evil", "trait": "evil", "coefficient": 2.5}
    assert cfg["logging_dir"] == "logs/t/werewolf_8p"
    assert runs["gbs_3p/gbs_3p__none.yaml"]["steering"] is None
    assert not any("gbs" in k and "evil_team" in k for k in runs)
    parsed = RunConfig.from_dict(cfg)
    assert parsed.players == 8 and parsed.episodes == 3 and parsed.vectors_dir.endswith("qwen3.5-9b/bf16")


def test_observation_axis_multiplies_runs():
    sweep = {**SWEEP, "games": {"c4": {"family": "symbolic", "id": "connect_four", "players": 2,
                                       "observation": ["text", "image"]}}}
    runs = expand(sweep)
    assert len(runs) == 2 * (1 + 2 * 2)  # 2 modes x (baseline + 2 traits x 2 coefficients); no team conditions
    assert runs["c4_2p/c4_2p__none__obs-image.yaml"]["observation"] == "image"
    assert runs["c4_2p/c4_2p__none.yaml"]["observation"] == "text"
