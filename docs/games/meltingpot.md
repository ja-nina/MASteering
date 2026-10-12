# Melting Pot

**Family:** `meltingpot` | **ID:** any Melting Pot substrate name, e.g. `commons_harvest__open`

DeepMind's pixel-based multi-agent social dilemmas (Agapiou et al., 2023), played by the
VLM from images.

## How it is played

Melting Pot steps every player simultaneously, hundreds of times per episode. An LLM call
per player per step is too slow and too fine-grained, so each agent answers with a **plan**
of up to `plan_length` primitive actions:

    ACTIONS: TURN_LEFT, FORWARD, FORWARD, FIRE_ZAP

The plan runs one action per step. Only agents whose plan has run out are asked again; the
others keep going, so the game advances without waiting for the slowest agent. The episode
ends when the substrate ends it or after `max_steps` steps. The action names are derived from
the substrate's action set (`NOOP`, `FORWARD`, `STEP_LEFT`, `TURN_RIGHT`, `FIRE_ZAP`,
`FIRE_CLEAN`, ...).

**Observation:** image only (`observation: image`). Each agent sees its own egocentric RGB view;
with `global_view: true` it also sees the whole map. The text part of the prompt carries the step,
the agent's reward so far, scalar observations (e.g. ready-to-shoot) and its last plan.

**Rewards:** summed per agent; the final step returns the episode totals. Melting Pot draws its
own randomness, so episodes are **not** paired across steering conditions (unlike the other games).

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `max_steps` | 200 | Episode cap in environment steps |
| `plan_length` | 4 | Maximum actions per answer |
| `global_view` | `false` | Also show the whole map (`WORLD.RGB`) |
| `python` | `$MELTINGPOT_PYTHON` | Run the substrate in a worker with this interpreter |

## Installing

`pip install dm-meltingpot` (Linux; needs `dmlab2d`, whose Python 3.12 wheels need glibc ≥ 2.35).
If it cannot go into the vLLM env, make a small separate env (Python 3.11, numpy, dm-meltingpot)
and set `MELTINGPOT_PYTHON=/path/to/that/python`. Each episode then runs its substrate in a
worker process of that interpreter (`python -m testbed.envs.meltingpot.backend`), talking to
the game runner over pipes. Substrate names starting with `fake:` run a tiny fake substrate
(used by the tests).

## Substrates in the main sweep

| Substrate | Players | Dilemma |
|---|---|---|
| `commons_harvest__open` | 7 | Over-harvesting apples kills regrowth |
| `clean_up` | 7 | Someone must clean the river for apples to grow |
| `prisoners_dilemma_in_the_matrix__repeated` | 2 | Repeated PD via collected resources |
| `stag_hunt_in_the_matrix__repeated` | 2 | Coordination: stag (risky, best) vs. hare (safe) |

`testbed/renderers/meltingpot.py` has short rule descriptions for these and a few more substrates.
Others get a generic description, so add one there before using them.
