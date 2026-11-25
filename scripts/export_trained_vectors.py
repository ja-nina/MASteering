"""Make one vectors_dir that holds PersVecFast's source vectors AND a PersVecTrain run's trained vectors under new
trait names, so one sweep can steer with both (and their random controls) side by side.

    python scripts/export_trained_vectors.py \
        --source  /scratch/inf0/user/nzukowsk/PersVecFast/out/persona/qwen3.5-2b/bf16 \
        --trained /scratch/inf0/user/nzukowsk/PersVecFast/out/trained/solo_kl10_t2048/persona/qwen3.5-2b/bf16 \
        --out     /scratch/inf0/user/nzukowsk/PersVecFast/out_games_kl10t2048/persona/qwen3.5-2b/bf16 \
        --suffix  kl10t2048

Result (testbed/steering.py's layout):
    <out>/<trait>.pt, .json, _raw.pt, ...            symlinks to every source file (the source traits, unchanged)
    <out>/vllm_hook/<file>                           symlinks to every source vllm_hook file
    <out>/<trait>-<suffix>.pt                        the trained run's {layer: vector} (its training layer replaced)
    <out>/<trait>-<suffix>.json                      the trained run's summary (source sweep summary + "trained")
    <out>/vllm_hook/<trait>-<suffix>_L<l>.pt         the trained vector, {"dir": v}
    <out>/vllm_hook/<trait>-<suffix>_L<l>_random.pt  its norm-matched random control (same directions as the source
                                                     controls, seed RANDOM_CONTROL_SEED + l, scaled to |trained|)
    <out>/export_manifest.json                       what was linked / copied, norms, file hashes
The <out> path should end in persona/<model>/bf16 under a PersVecFast-like root, so the testbed's default traits_dir
(<out>/../../../data/persona_vectors) is still found; or set traits_dir in the sweep. Never writes into --source
or --trained; refuses an existing --out unless --force.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


def sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def trained_traits(trained: Path) -> list[str]:
    """Traits of the run: <trait>.pt files that have a vllm_hook/<trait>_L<l>.pt (written last but one)."""
    out = []
    for p in sorted(trained.glob("*.pt")):
        if p.stem.endswith("_raw"):
            continue
        if list((trained / "vllm_hook").glob(f"{p.stem}_L*.pt")):
            out.append(p.stem)
    return out


def trained_layer(trained: Path, trait: str) -> int:
    meta = json.loads((trained / f"{trait}.json").read_text(encoding="utf-8"))
    if "trained" not in meta:
        raise SystemExit(f"{trained / (trait + '.json')} has no 'trained' entry: not a PersVecTrain run dir")
    return int(meta["trained"]["layer"])


def export(source: Path, trained: Path, out: Path, suffix: str, force: bool = False) -> dict:
    source, trained, out = Path(source).resolve(), Path(trained).resolve(), Path(out)
    for p in (source, trained):
        if not p.is_dir():
            raise SystemExit(f"{p} is not a directory")
    if out.resolve() in (source, trained) or source in out.resolve().parents or trained in out.resolve().parents:
        raise SystemExit(f"--out {out} must not be inside --source or --trained")
    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists (use --force to rebuild it)")
        shutil.rmtree(out)
    (out / "vllm_hook").mkdir(parents=True)

    linked = 0
    for p in sorted(source.iterdir()):
        if p.is_file():
            os.symlink(p, out / p.name)
            linked += 1
    for p in sorted((source / "vllm_hook").iterdir()) if (source / "vllm_hook").is_dir() else []:
        os.symlink(p, out / "vllm_hook" / p.name)
        linked += 1

    traits = trained_traits(trained)
    if not traits:
        raise SystemExit(f"no trained traits in {trained}")
    manifest = {"source": str(source), "trained": str(trained), "suffix": suffix, "linked_source_files": linked,
                "traits": {}}
    for t in traits:
        new = f"{t}-{suffix}"
        if (source / f"{new}.pt").exists():
            raise SystemExit(f"{new}.pt already exists in the source dir; pick another --suffix")
        layer = trained_layer(trained, t)
        hook = trained / "vllm_hook" / f"{t}_L{layer}.pt"
        hook_rnd = trained / "vllm_hook" / f"{t}_L{layer}_random.pt"
        v = torch.load(hook, map_location="cpu")["dir"].float()
        per_layer = torch.load(trained / f"{t}.pt", map_location="cpu")
        if not torch.allclose(per_layer[layer].float(), v, atol=1e-3):
            raise SystemExit(f"{t}: {t}.pt layer {layer} differs from vllm_hook/{hook.name}")
        src_v = torch.load(source / f"{t}.pt", map_location="cpu")[layer].float()
        shutil.copy2(trained / f"{t}.pt", out / f"{new}.pt")
        shutil.copy2(trained / f"{t}.json", out / f"{new}.json")
        shutil.copy2(hook, out / "vllm_hook" / f"{new}_L{layer}.pt")
        if hook_rnd.exists():
            shutil.copy2(hook_rnd, out / "vllm_hook" / f"{new}_L{layer}_random.pt")
        manifest["traits"][new] = {
            "from": t, "layer": layer, "trained_norm": float(v.norm()), "source_norm": float(src_v.norm()),
            "cos_with_source": float(v @ src_v / (v.norm() * src_v.norm())),
            "sha1": {"pt": sha(out / f"{new}.pt"), "hook": sha(out / "vllm_hook" / f"{new}_L{layer}.pt")},
            "random_control": hook_rnd.exists()}
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_traits(traits_src: Path, trained: Path, traits_out: Path, suffix: str, force: bool = False) -> list[str]:
    """A traits_dir for prompt personas: symlinks to every PersVecFast trait definition, plus <trait>-<suffix>.json =
    a copy of <trait>.json for each trained trait (the trained twin's persona prompt is its source trait's)."""
    traits_src, trained, traits_out = Path(traits_src).resolve(), Path(trained).resolve(), Path(traits_out)
    if traits_out.resolve() == traits_src or traits_src in traits_out.resolve().parents:
        raise SystemExit(f"--traits-out {traits_out} must not be inside --traits-src")
    if traits_out.exists():
        if not force:
            raise SystemExit(f"{traits_out} exists (use --force to rebuild it)")
        shutil.rmtree(traits_out)
    traits_out.mkdir(parents=True)
    for p in sorted(traits_src.iterdir()):
        if p.is_file():
            os.symlink(p, traits_out / p.name)
    made = []
    for t in trained_traits(trained):
        if not (traits_src / f"{t}.json").exists():
            raise SystemExit(f"no trait definition {traits_src / (t + '.json')}")
        shutil.copy2(traits_src / f"{t}.json", traits_out / f"{t}-{suffix}.json")
        made.append(f"{t}-{suffix}")
    return made


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits-src", dest="traits_src", help="PersVecFast/data/persona_vectors (with --traits-out)")
    ap.add_argument("--traits-out", dest="traits_out", help="also write a traits_dir for prompt personas here")
    ap.add_argument("--traits-only", action="store_true", dest="traits_only",
                    help="only the traits_dir (an existing --out vectors dir is not touched)")
    ap.add_argument("--source", help="PersVecFast vectors dir (.../persona/<model>/bf16)")
    ap.add_argument("--trained", required=True, help="PersVecTrain run vectors dir (.../trained/<run>/persona/<model>/bf16)")
    ap.add_argument("--out", help="new vectors dir, ending in persona/<model>/bf16")
    ap.add_argument("--suffix", required=True, help="trained traits become <trait>-<suffix>")
    ap.add_argument("--force", action="store_true", help="rebuild an existing --out / --traits-out")
    a = ap.parse_args(argv)
    if bool(a.traits_src) != bool(a.traits_out):
        raise SystemExit("--traits-src and --traits-out go together")
    if a.traits_out:
        made = export_traits(Path(a.traits_src), Path(a.trained), Path(a.traits_out), a.suffix, a.force)
        print(f"traits_dir {a.traits_out}: source definitions linked, {len(made)} trained twins: {', '.join(made)}")
    if a.traits_only:
        return
    if not (a.source and a.out):
        raise SystemExit("--source and --out are needed (or --traits-only)")
    m = export(Path(a.source), Path(a.trained), Path(a.out), a.suffix, a.force)
    print(f"linked {m['linked_source_files']} source files; trained traits:")
    for k, v in m["traits"].items():
        print(f"  {k:28} L{v['layer']}  |v| {v['trained_norm']:.3f} (source {v['source_norm']:.3f})  "
              f"cos {v['cos_with_source']:.3f}  random {'yes' if v['random_control'] else 'MISSING'}")
    print(f"wrote {Path(a.out) / 'export_manifest.json'}")


if __name__ == "__main__":
    main()
