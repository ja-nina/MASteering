"""scripts/export_trained_vectors.py: the merged dir steers source traits with the source vectors and <trait>-<suffix>
with the trained ones, through the testbed's own VectorStore."""
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from testbed.steering import VectorStore, random_control

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("etv", REPO / "scripts" / "export_trained_vectors.py")
etv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(etv)


def _can_symlink(tmp_path):
    try:
        os.symlink(tmp_path / "a", tmp_path / "b")
        return True
    except (OSError, NotImplementedError):
        return False


def _make(tmp_path):
    torch.manual_seed(0)
    src = tmp_path / "pvf" / "out" / "persona" / "m" / "bf16"
    trn = tmp_path / "pvf" / "out" / "trained" / "run" / "persona" / "m" / "bf16"
    for d in (src / "vllm_hook", trn / "vllm_hook"):
        d.mkdir(parents=True)
    vs = {l: torch.randn(8) for l in range(4)}
    torch.save(vs, src / "evil.pt")
    (src / "evil.json").write_text(json.dumps({"layers": [1, 2]}))
    torch.save({"dir": vs[2]}, src / "vllm_hook" / "evil_L2.pt")
    t = torch.randn(8)
    torch.save({**vs, 2: t}, trn / "evil.pt")
    (trn / "evil.json").write_text(json.dumps({"layers": [1, 2], "trained": {"layer": 2}}))
    torch.save({"dir": t}, trn / "vllm_hook" / "evil_L2.pt")
    torch.save({"dir": random_control(t, 2)}, trn / "vllm_hook" / "evil_L2_random.pt")
    return src, trn, vs, t


def test_merged_dir_steers_both(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("no symlink permission here")
    src, trn, vs, t = _make(tmp_path)
    out = tmp_path / "pvf" / "out_games" / "persona" / "m" / "bf16"
    m = etv.export(src, trn, out, "kl10")
    assert set(m["traits"]) == {"evil-kl10"}
    store = VectorStore(out, tmp_path / "cache")
    assert set(store.traits()) == {"evil", "evil-kl10"}
    assert torch.allclose(torch.load(store.steer_file({"evil": 1.0}, 2))["dir"], vs[2])
    assert torch.allclose(torch.load(store.steer_file({"evil-kl10": 1.0}, 2))["dir"], t)
    rnd = torch.load(store.steer_file({"evil-kl10": 1.0}, 2, random_ctrl=True))["dir"]
    assert torch.allclose(rnd.norm(), t.norm(), atol=1e-5)
    assert store.candidate_layers() == [1, 2]
    with pytest.raises(SystemExit):
        etv.export(src, trn, out, "kl10")  # exists, no --force
