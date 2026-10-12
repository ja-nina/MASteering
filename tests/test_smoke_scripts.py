"""The cluster smoke scripts, run on CPU with fake engines: they finish and save their samples."""
import html
import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts import run_games
from scripts.collect_samples import main as collect
from scripts.make_configs import expand, main as make_configs
from tests.fakes import FakeGenerator, FakeReader, fake_open_engines, make_vectors

ROOT = Path(__file__).resolve().parents[1]


def _sweep(name):
    return yaml.safe_load((ROOT / "config" / "sweeps" / f"{name}.yaml").read_text(encoding="utf-8"))


def test_smoke_sweep_covers_every_game_of_the_main_sweep():
    smoke, main = _sweep("smoke"), _sweep("main")
    assert {(g["family"], g["id"]) for g in main["games"].values()} <= \
        {(g["family"], g["id"]) for g in smoke["games"].values()}
    from testbed.registry import _SYMBOLIC
    assert set(_SYMBOLIC) <= {g["id"] for g in smoke["games"].values() if g["family"] == "symbolic"}
    vias = {(c or {}).get("via", "vector") for c in smoke["conditions"].values() if c}
    assert vias == {"vector", "prompt", "both"} and None in smoke["conditions"].values()
    runs = expand(smoke)
    for g in smoke["games"].values():  # every image-capable game is smoke-tested in image mode too
        modes = g.get("observation", "text")
        assert any(r["game"]["id"] == g["id"] and r["observation"] == m
                   for r in runs.values() for m in (modes if isinstance(modes, list) else [modes]))


def _pvf(tmp_path):
    """A PersVecFast checkout stand-in: out/persona/<model>/bf16 + data/persona_vectors."""
    vec = make_vectors(tmp_path / "PersVecFast" / "out" / "persona" / "qwen3.5-9b" / "bf16")
    (tmp_path / "PersVecFast" / "data").mkdir()
    (vec.parent / "traits").rename(tmp_path / "PersVecFast" / "data" / "persona_vectors")
    return tmp_path / "PersVecFast" / "out"


def test_symbolic_smoke_games_and_samples_page(tmp_path, monkeypatch):
    monkeypatch.setenv("PVF_OUT", str(_pvf(tmp_path)))
    sweep = _sweep("smoke")
    sweep["games"] = {k: g for k, g in sweep["games"].items() if g["family"] == "symbolic"}
    sweep["logging_dir"] = str(tmp_path / "logs")
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(sweep))
    make_configs([str(tmp_path / "s.yaml"), "--out", str(tmp_path / "runs")])
    assert run_games.main(["--config-dir", str(tmp_path / "runs")], open_engines=fake_open_engines()) in (0, 1)
    out = tmp_path / "samples.html"
    collect([str(tmp_path / "logs"), "--configs", str(tmp_path / "runs"), "--out", str(out)])
    page = out.read_text(encoding="utf-8")
    status = list(__import__("csv").DictReader(out.with_suffix(".csv").open(encoding="utf-8")))
    assert len(status) == len(list((tmp_path / "runs").rglob("*.yaml")))
    ok = [r for r in status if r["status"] == "ok"]
    assert len(ok) >= 0.9 * len(status)  # fake replies are random: a few games may fall back a lot
    assert "data:image/png;base64," in page  # image runs embed their saved images
    assert html.escape("persona prompt: You clearly exhibit the trait 'sarcastic'.") in page
    assert "prompt(sarcastic)+sarcastic@L2x2.5" in page  # via: both


def test_engine_smoke_saves_samples(tmp_path, monkeypatch):
    pvf_src = ROOT.parent / "PersVecFast" / "src"
    if pvf_src.exists():
        sys.path.insert(0, str(pvf_src))
    pytest.importorskip("persvecfast")
    from scripts import smoke_engines
    monkeypatch.setenv("PVF_OUT", str(_pvf(tmp_path)))
    gen = FakeGenerator(respond=lambda r: "a reply " * 5)
    monkeypatch.setattr(smoke_engines, "open_engines", fake_open_engines(gen))
    monkeypatch.chdir(tmp_path)
    smoke_engines.main(["--out", str(tmp_path / "eng")])  # fake rows: the "raises" checks fail; that is fine here
    samples = json.loads((tmp_path / "eng" / "samples.json").read_text(encoding="utf-8"))
    conds = {s["condition"] for s in samples["text"]}
    assert conds == {"none", "up", "down", "zero", "prompt"} and len(samples["image"]) == 8
    prompt = next(s for s in samples["text"] if s["condition"] == "prompt")
    assert prompt["system"].startswith("You clearly exhibit the trait 'sarcastic'.")
    assert {c["check"] for c in samples["checks"]} >= {"image prompts generate"}
    assert (tmp_path / "eng" / "board.png").exists()
