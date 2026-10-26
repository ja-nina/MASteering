import json

import pytest

torch = pytest.importorskip("torch")


def test_vector_norms_per_trait_and_layer_and_exported_random(tmp_path):
    from scripts.vector_norms import main
    vec = tmp_path / "vectors"
    (vec / "vllm_hook").mkdir(parents=True)
    torch.save({16: torch.tensor([3.0, 4.0]), 32: torch.tensor([6.0, 8.0])}, vec / "evil.pt")
    torch.save({"dir": torch.tensor([0.0, 7.0])}, vec / "vllm_hook" / "evil_L32_random.pt")
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(f"name: s\nmodel: m\nvectors_dir: {vec.as_posix()}\ngames: {{}}\n", encoding="utf-8")
    out = tmp_path / "norms.json"
    assert main(["--sweep", str(sweep), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["norms"] == {"evil": {"16": 5.0, "32": 10.0}}
    assert data["random"] == {"evil": {"32": 7.0}}
