# LLM Steering Multi-Game Testbed

LLM agents play text-based multi-agent games while some of them are **steered with persona
vectors**, and every turn is **monitored** by projecting the completion onto every persona
vector. Generation runs on [vLLM-Hook](https://github.com/IBM/vLLM-Hook), reusing
[PersVecFast](../PersVecFast)'s model specs, sampling settings and vectors, so a game turn
costs about as much as a plain vLLM request.

## How it works

```
config/sweeps/main.yaml ──make_configs.py──▶ config/runs/main/<game>_<n>p/<run_id>.yaml
                                                   │  (one SLURM array per game group)
                                                   ▼
scripts/run_games.py ── lockstep scheduler (testbed/runner.py) ──────────────────────────
   every step: the pending turns of up to `concurrent_episodes` episodes, of all runs
   in the group ──▶ ONE generate call on the steer engine
                    (each request carries its own add_vector config; unsteered = coefficient 0)
                ──▶ ONE readout on the capture engine (prompt + completion prefill,
                    monitor layers) ──▶ projections onto every trait vector
                ──▶ parse (re-ask with feedback on errors) ──▶ env.submit ──▶ JSONL logs
```

- **Steer engine** (vLLM-Hook worker `steer`): hidden state += `coefficient · v` at one decoder
  block, with `v` the raw PersVecFast mean-difference vector. This is the same op and scale
  PersVecFast evaluated (coefficients 1.25 / 2.5), and it stays inside vLLM's CUDA graphs.
  Prefix caching is on, so the long shared game prompts are cheap.
- **Capture engine** (worker `capture_hs`), live next to it, on a second GPU or sharing one.
  vLLM-Hook cannot capture in a steering engine, so the monitor is a **readout by the
  unsteered model** of what each agent wrote: mean post-block residual over the completion's
  tokens, projected onto each unit persona vector, at PersVecFast's candidate layers. For a
  steered agent this measures the trait in its text as the base model reads it, not the
  vector that was injected.
- **Paired conditions:** an episode's seed depends only on (game, players, episode), so every
  steering condition plays the same deals, roles and seat draws (except Melting Pot).
- **Text or image observations:** the models are vision-language models, so a run's
  `observation: text | image | both` shows the game state as text, as a picture, or both
  ([observations.md](docs/games/observations.md)). Melting Pot is pixel-only.

## Steering conditions

A run's `steering:` is one group or a list of groups (`testbed/steering.py`). A group injects a
trait through one of two channels, or both (`via`):

- `vector`: the extracted persona vector added to the residual stream (vLLM-Hook `add_vector`)
- `prompt`: the trait's PersVecFast instruction (the exact prompt the vector was extracted with)
  put before the agent's system prompt; negative coefficient = the negative instruction
- `both`: both at once

All three run on the same vLLM-Hook steer engine (prompt-only requests carry coefficient 0).

```yaml
steering:
  - team: evil            # pool: everyone, narrowed by team (good|evil), roles, agents (ids or seats)
    select: all           # all | one | count (+count) | fraction (+fraction); drawn per episode
    trait: evil           # or traits: {evil: 1.0, sarcastic: 0.5}   (weighted mix)
    coefficient: 2.5      # negative = suppress
    layer: auto           # the trait's best PersVecFast layer, or an int (0-based block)
  - team: good
    select: one
    trait: empathy
    coefficient: 2.5
    random_control: false # true: a random direction with the vector's norm
    via: vector           # vector | prompt | both
```

The main sweep (`config/sweeps/main.yaml`) crosses games × traits × coefficients with the
conditions `none`, `one`, `half`, `all`, `all_random` and, for hidden-role games, `evil_team`,
`evil_one`, `good_one`, and the prompt-persona conditions `prompt_one`, `prompt_all`, `both_one`,
`evil_team_prompt`.

## Games

| Family | ID | Players | Notes | Doc |
|---|---|---|---|---|
| symbolic | `gbs` | 2+ | Goldstone group sum; exact / directional / picking variants | [gbs.md](docs/games/gbs.md) |
| symbolic | `beauty_contest` | 2+ | Keynesian 2/3-of-average | [beauty_contest.md](docs/games/beauty_contest.md) |
| symbolic | `harvest` | 2+ | Commons harvest (GovSim fishery) with discussion | [harvest.md](docs/games/harvest.md) |
| symbolic | `werewolf` | 4–15 | Hidden roles; teams for steering | [werewolf.md](docs/games/werewolf.md) |
| symbolic | `avalon` | 5–10 | Hidden roles; teams for steering | [avalon.md](docs/games/avalon.md) |
| symbolic | `hanabi` | 2–5 | Cooperative | [hanabi.md](docs/games/hanabi.md) |
| symbolic | `overcooked` | 2 | Cooperative kitchen | [overcooked.md](docs/games/overcooked.md) |
| symbolic | `tictactoe`, `connect_four`, `gomoku` | 2–4 | k-in-a-row board games | [board_games.md](docs/games/board_games.md) |
| meltingpot | any substrate | varies | Pixel social dilemmas (commons harvest, clean up, …); image only | [meltingpot.md](docs/games/meltingpot.md) |
| textarena | any `env_id` | varies | SecretMafia (teams), Debate, IPD, Negotiation, … | [textarena.md](docs/games/textarena.md) |

Every symbolic game has text, image and both modes; TextArena games are text only.

## Setup (cluster)

Use the PersVecFast env (vLLM 0.29, vLLM-Hook, persvecfast), plus this repo's requirements:

    micromamba activate pvf
    pip install -e /scratch/inf0/user/nzukowsk/PersVecFast
    pip install -r requirements.txt
    cp .env.example .env            # OPENROUTER_API_KEY, for TextArena juries only
    pip install dm-meltingpot       # optional, for Melting Pot (or MELTINGPOT_PYTHON, see meltingpot.md)

Vectors are read from `${PVF_OUT}/persona/<model>/bf16` (default
`PVF_OUT=/scratch/inf0/user/nzukowsk/PersVecFast/out`); override per run with `vectors_dir:`.

## Run

    # 0. on the login node, no GPU: every game of a sweep builds and resets
    python scripts/check_games.py config/sweeps/smoke.yaml

    # 1. cluster smoke tests (two jobs, 2 GPUs each):
    #    engine checks: vector / prompt persona steering moves the monitor, coefficient 0 ==
    #      unsteered, image prompts work            -> results/smoke/engines/<model>/samples.json
    #    every game x {none, prompt, vector, both, random, team} x {text, image}, 1 episode each
    #                                               -> results/smoke/samples.html (+ samples.csv status)
    bash scripts/slurm/submit_smoke.sh qwen3.5-9b

    # 3. the main sweep: one array per game group, one task per ~400 episodes (DRY_RUN=1 to preview)
    python scripts/make_configs.py config/sweeps/main.yaml
    bash scripts/slurm/submit_sweep.sh config/runs/main

    # progress while it runs: per game (and per task with --tasks), from the SLURM logs
    python scripts/progress.py --configs config/runs/main

    # 4. results: one row per agent per episode (condition, team, reward, monitor means), then the
    #    analysis report over all games (coverage, does steering move the trait, outcome and
    #    spillover effects per game x trait, evil-team steering): one HTML page
    python scripts/summarize_runs.py <logging_dir>/main [<fallback>/main] --out results/main/agents.csv.gz
    python scripts/report.py results/main/agents.csv.gz --out results/main/report.html
    # the same in plain terms (no slopes or SDs: words, win rates, markers; per game, vector and setting)
    python scripts/steering_mini_report.py results/main/agents.csv.gz --out results/main/steering_mini_report.html
    # (a .csv.gz past 90 MB continues in agents-2.csv.gz, ...; the report reads all parts)
    # or both as a job (a large sweep needs more memory than the submit node allows):
    sbatch scripts/slurm/report.sbatch main

Resubmitting resumes (episodes with a summary file are skipped). One episode, locally on a GPU:
`python scripts/run_games.py --config <run config> --episodes 1`. Samples page for any sweep:
`python scripts/collect_samples.py logs/<sweep> --configs config/runs/<sweep> --out results/<sweep>/samples.html`
(one self-contained HTML: run status, steering incl. persona prompt, sample turns with images).

## Outputs

`logs/<sweep>/<game>_<n>p/<run_id>/`
- `run.json`: the run config, with each steering group's resolved layer, coefficient and vector file
- `episode_N.jsonl`: one line per agent per turn, with everything the agent was given and said:
  - `system_prompt`: the system prompt, with the persona prompt prepended for prompt-steered agents
  - `user_prompt`: the user turn of the final attempt
  - `completion`: the final output
  - `attempts`: every attempt in order (its user prompt including the parser's feedback, its output,
    its parse error)
  - `model_input`: the exact chat-templated text the model received
  - plus the parsed action, fallback flag, reward, `steering` (null if unsteered) and `monitor`
    `{layer: {trait: projection}}`
- `episode_N.trace.txt`: the same as a readable transcript, including rejected attempts
- `episode_N_images/`: each turn's image observations (with `save_images: true`)
- `episode_N.summary.json`: final rewards, roles, teams, eliminations, who was steered with what

## Layout

```
testbed/
  steering.py      VectorStore (PersVecFast output), SteerGroup, SteeringPlan
  engine.py        TurnRequest / TurnResult, Monitor (projection onto unit trait vectors)
  visual.py        observation modes (text | image | both) and PIL drawing helpers
  mia_engine.py    vLLM-Hook steer + capture engines (only module importing vllm)
  runner.py        lockstep scheduler over concurrent episodes
  config.py        run config, episode seeds, sharding, job building
  registry.py      game id -> (adapter, renderer, parser)
  envs/ renderers/ parsers/   games (envs/meltingpot: substrate backend + plan adapter)
  logging_/        JSONL + trace + summary writer
scripts/
  make_configs.py  sweep -> run configs      run_games.py   play
  summarize_runs.py logs -> CSV             smoke_engines.py   GPU engine checks (+ samples)
  report.py        CSV -> analysis report   progress.py        sweep progress from SLURM logs
  steering_mini_report.py CSV -> the report in plain terms (examples, settings, synergy)
  collect_samples.py logs -> samples page (HTML + status CSV)
  slurm/           env.sh (shared setup), run_games.sbatch, submit_sweep.sh,
                   smoke.sbatch, smoke_games.sbatch, submit_smoke.sh
config/sweeps/     main.yaml, smoke.yaml
```

## Tests (no GPU)

    python -m pytest -q

`tests/fakes.py` stands in for both engines and a PersVecFast vector directory, so the runner,
steering assignment, monitoring, sharding/resume and all games are tested on CPU.
