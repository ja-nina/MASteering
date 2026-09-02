"""Aggregate per-checkpoint KL JSON files produced by the array sweep.

Usage:
    python scripts/aggregate_kl.py /scratch/.../kl_results/ \
        --output /scratch/.../kl_calibration.json
"""
import argparse, json, math, sys
from pathlib import Path


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    results = []
    for f in sorted(args.results_dir.glob("kl_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        # each file contains {"args": ..., "results": [...]} from measure_kl
        if isinstance(data, dict) and "results" in data:
            results.extend(data["results"])
        elif isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)

    valid = [r for r in results if "mean_kl_nats" in r]
    errors = [r for r in results if "error" in r]

    print(f"Total checkpoints : {len(results)}")
    print(f"  successful      : {len(valid)}")
    print(f"  errors          : {len(errors)}")

    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  {Path(r['lora_dir']).name}: {r['error']}")

    if valid:
        kls  = [r["mean_kl_nats"] for r in valid]
        ppls = [r["mean_base_ppl"] for r in valid]
        ttrs = [r["mean_ttr"] for r in valid]

        kls_sorted = sorted(kls)
        median_kl  = kls_sorted[len(kls_sorted) // 2]
        target     = round(median_kl * 1.1, 1)

        print(f"\n{'═'*60}")
        print("CALIBRATION SUMMARY")
        print(f"{'═'*60}")
        print(f"  KL range          : {min(kls):.3f} – {max(kls):.3f} nats")
        print(f"  KL mean ± std     : {sum(kls)/len(kls):.3f} ± {_std(kls):.3f} nats")
        print(f"  KL median         : {median_kl:.3f} nats")
        print(f"  Base PPL range    : {min(ppls):.1f} – {max(ppls):.1f}")
        print(f"  TTR range         : {min(ttrs):.3f} – {max(ttrs):.3f}")
        print()
        print(f"  RECOMMENDED kl_target  : {target:.1f} nats")
        print(f"  RECOMMENDED initial β  : {max(0.01, round(0.5 / target, 3))}")
        print(f"  Adaptive rule: if KL > 1.5×target → β×=2; if KL < 0.5×target → β÷=2")
        print(f"{'═'*60}")

        print(f"\nTop 10 highest KL (most drift):")
        for r in sorted(valid, key=lambda x: -x["mean_kl_nats"])[:10]:
            print(f"  {Path(r['lora_dir']).name:<50} {r['mean_kl_nats']:.4f} nats")

        print(f"\nTop 10 lowest KL (least drift):")
        for r in sorted(valid, key=lambda x: x["mean_kl_nats"])[:10]:
            print(f"  {Path(r['lora_dir']).name:<50} {r['mean_kl_nats']:.4f} nats")

    combined = {"results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved to: {args.output}")
    else:
        print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
