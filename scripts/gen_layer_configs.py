"""Generate per-layer activation-steering experiment configs.

After running the layer sweep, call this once to produce one YAML per layer
for every (game, scope) combination you want to evaluate:

  python scripts/gen_layer_configs.py
  python scripts/gen_layer_configs.py --start-layer 10 --end-layer 35 \\
      --stream resid --top-n 16 --d-sae 16384 --k 32 --coefficient 20.0

Output: config/layer_sweep/{game}_exp_activation_{scope}_l{N}.yaml
"""
from __future__ import annotations

import argparse
import os

TEMPLATE_BEAUTY_ALL = """\
run_id: {run_id}

game:
  family: symbolic
  id: beauty_contest
  env_kwargs:
    num_rounds: 5
episodes: 50

model:
  backend: transformers
  model_id: Qwen/Qwen3-4B
  temperature: 0.7
  top_p: 0.8
  top_k: 20

agents:
  count: 4
  concurrency: sequential
  max_parse_retries: 5

steering:
  default: activation
  default_config:
    layer: {layer}
    vector_path: {vector_path}
    coefficient: {coefficient}
  per_agent: {{}}

logging:
  dir: logs/

wandb:
  enabled: true
  project: ma-steering
  name: {run_id}
  tags: [beauty_contest, activation, sae, all_agents, layer_{layer_idx}]
"""

TEMPLATE_BEAUTY_ONE = """\
run_id: {run_id}

game:
  family: symbolic
  id: beauty_contest
  env_kwargs:
    num_rounds: 5
episodes: 50

model:
  backend: transformers
  model_id: Qwen/Qwen3-4B
  temperature: 0.7
  top_p: 0.8
  top_k: 20

agents:
  count: 4
  concurrency: sequential
  max_parse_retries: 5

steering:
  default: activation
  per_agent:
    player_0:
      layer: {layer}
      vector_path: {vector_path}
      coefficient: {coefficient}

logging:
  dir: logs/

wandb:
  enabled: true
  project: ma-steering
  name: {run_id}
  tags: [beauty_contest, activation, sae, one_agent, layer_{layer_idx}]
"""

TEMPLATE_GBS_ALL = """\
run_id: {run_id}

game:
  family: symbolic
  id: gbs
  env_kwargs:
    num_rounds: 10
episodes: 50

model:
  backend: transformers
  model_id: Qwen/Qwen3-4B
  temperature: 0.7
  top_p: 0.8
  top_k: 20

agents:
  count: 4
  concurrency: sequential
  max_parse_retries: 5

steering:
  default: activation
  default_config:
    layer: {layer}
    vector_path: {vector_path}
    coefficient: {coefficient}
  per_agent: {{}}

logging:
  dir: logs/

wandb:
  enabled: true
  project: ma-steering
  name: {run_id}
  tags: [gbs, activation, sae, all_agents, layer_{layer_idx}]
"""

TEMPLATE_GBS_ONE = """\
run_id: {run_id}

game:
  family: symbolic
  id: gbs
  env_kwargs:
    num_rounds: 10
episodes: 50

model:
  backend: transformers
  model_id: Qwen/Qwen3-4B
  temperature: 0.7
  top_p: 0.8
  top_k: 20

agents:
  count: 4
  concurrency: sequential
  max_parse_retries: 5

steering:
  default: activation
  per_agent:
    player_0:
      layer: {layer}
      vector_path: {vector_path}
      coefficient: {coefficient}

logging:
  dir: logs/

wandb:
  enabled: true
  project: ma-steering
  name: {run_id}
  tags: [gbs, activation, sae, one_agent, layer_{layer_idx}]
"""

VARIANTS = [
    ("beauty_contest", "all",  TEMPLATE_BEAUTY_ALL),
    ("beauty_contest", "one",  TEMPLATE_BEAUTY_ONE),
    ("gbs",            "all",  TEMPLATE_GBS_ALL),
    ("gbs",            "one",  TEMPLATE_GBS_ONE),
]


def vector_path(game: str, layer_idx: int, stream: str,
                top_n: int, d_sae: int, k: int, vectors_dir: str) -> str:
    layer_tag = f"model_layers_{layer_idx}"
    sae_stem = f"{game}_{layer_tag}_{stream}_d{d_sae}_k{k}"
    return os.path.join(vectors_dir, f"tom_sae_top{top_n}_{sae_stem}.npy")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-layer", type=int, default=10)
    ap.add_argument("--end-layer",   type=int, default=35)
    ap.add_argument("--stream",      default="resid", choices=["resid", "mlp"],
                    help="Which stream's vector to use for steering (default: resid)")
    ap.add_argument("--top-n",       type=int, default=16)
    ap.add_argument("--d-sae",       type=int, default=16384)
    ap.add_argument("--k",           type=int, default=32)
    ap.add_argument("--coefficient", type=float, default=20.0)
    ap.add_argument("--vectors-dir", default="vectors")
    ap.add_argument("--output-dir",  default="config/layer_sweep")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generated = []
    for layer_idx in range(args.start_layer, args.end_layer + 1):
        layer = f"model.layers.{layer_idx}"
        for game, scope, template in VARIANTS:
            prefix = "" if game == "beauty_contest" else "gbs_"
            run_id = f"{prefix}exp_activation_{scope}_l{layer_idx}"
            vec = vector_path(game, layer_idx, args.stream,
                              args.top_n, args.d_sae, args.k, args.vectors_dir)
            content = template.format(
                run_id=run_id,
                layer=layer,
                layer_idx=layer_idx,
                vector_path=vec,
                coefficient=args.coefficient,
            )
            out = os.path.join(args.output_dir, f"{run_id}.yaml")
            with open(out, "w") as f:
                f.write(content)
            generated.append(out)

    print(f"Generated {len(generated)} configs in {args.output_dir}/")
    print(f"  layers {args.start_layer}-{args.end_layer}, stream={args.stream}, "
          f"top_n={args.top_n}, d_sae={args.d_sae}, k={args.k}, "
          f"coefficient={args.coefficient}")
    print()
    print("Run all beauty_contest/all evals:")
    print(f"  for f in {args.output_dir}/exp_activation_all_l*.yaml; do "
          "python scripts/run_episode.py --config \"$f\"; done")
    print()
    print("Or submit as a SLURM array — see scripts/run_evals.slurm")


if __name__ == "__main__":
    main()
