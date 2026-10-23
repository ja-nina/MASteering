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


def test_prompt_conditions_get_one_run_per_sign():
    sweep = {**SWEEP, "coefficients": [-2, -1, 1, 2], "traits": ["evil"],
             "conditions": {"prompt_one": {"select": "one", "via": "prompt"}, "one": {"select": "one"}},
             "games": {"gbs": {"family": "symbolic", "id": "gbs", "players": 3}}}
    runs = expand(sweep)
    prompt = sorted(k for k in runs if "prompt_one" in k)
    assert prompt == ["gbs_3p/gbs_3p__prompt_one__evil_x-1.yaml", "gbs_3p/gbs_3p__prompt_one__evil_x1.yaml"]
    assert sum("__one__" in k for k in runs) == 4


def test_load_sweep_with_another_model(tmp_path):
    from scripts.make_configs import load_sweep
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(SWEEP))
    assert load_sweep(path)["name"] == "t"
    assert load_sweep(path, "qwen3.5-9b")["name"] == "t"  # the sweep's own model: unchanged
    other = load_sweep(path, "qwen3.5-2b")
    assert (other["name"], other["model"]) == ("t_qwen3.5-2b", "qwen3.5-2b")
    cfg = next(iter(expand(other).values()))
    assert cfg["model"] == "qwen3.5-2b" and cfg["logging_dir"].startswith("logs/t_qwen3.5-2b/")


def test_log_name_extends_another_sweeps_logs(tmp_path):
    """A sweep with `log_name` logs into that sweep's dirs (same run_ids: finished episodes are kept), while its
    configs stay under its own name; another model suffixes both."""
    from scripts.make_configs import load_sweep
    sweep = {**SWEEP, "name": "focus", "log_name": "t", "episodes": 50}
    cfg = expand(sweep)["werewolf_8p/werewolf_8p__evil_team__evil_x2.5.yaml"]
    assert cfg["logging_dir"] == "logs/t/werewolf_8p" and cfg["episodes"] == 50
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(sweep))
    other = load_sweep(path, "qwen3.5-2b")
    assert (other["name"], other["log_name"]) == ("focus_qwen3.5-2b", "t_qwen3.5-2b")
    assert next(iter(expand(other).values()))["logging_dir"].startswith("logs/t_qwen3.5-2b/")


def test_model_override_narrows_grid_and_drops_monitor(tmp_path):
    """models.<key> may narrow coefficients (same run_ids for the kept cells) and switch the monitor off."""
    from scripts.make_configs import load_sweep
    sweep = {**SWEEP, "monitor": {"layers": "auto"},
             "models": {"qwen3.5-2b": {"coefficients": [2.5], "monitor": None, "save_images": True}}}
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(sweep))
    runs = expand(load_sweep(path, "qwen3.5-2b"))
    assert "gbs_3p/gbs_3p__all__evil_x2.5.yaml" in runs and not any("x1.25" in k for k in runs)
    cfg = runs["gbs_3p/gbs_3p__all__evil_x2.5.yaml"]
    assert cfg["monitor"] is None and cfg["save_images"] is True
    assert len(expand(load_sweep(path))) > len(runs)  # the sweep's own model keeps the full grid


def test_model_adds_conditions_of_its_own(tmp_path):
    from scripts.make_configs import load_sweep
    sweep = {**SWEEP, "models": {"qwen3.5-2b": {"conditions": {"all_L7": {"select": "all", "layer": 7}}}}}
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(sweep))
    runs = expand(load_sweep(path, "qwen3.5-2b"))
    assert runs["gbs_3p/gbs_3p__all_L7__evil_x2.5.yaml"]["steering"]["layer"] == 7
    assert "gbs_3p/gbs_3p__all__evil_x2.5.yaml" in runs  # the sweep's own conditions stay
    assert not any("all_L7" in k for k in expand(load_sweep(path)))  # other models: unchanged
