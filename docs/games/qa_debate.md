# Accuracy debate (`qa_debate`)

N agents answer the same question over a few rounds (default 3), following Du et al. (2023), *Improving
factuality and reasoning in language models through multiagent debate*. Unlike the other games, it has a right
answer, so steering is scored on a task instead of a payoff.

- **Round 1:** every agent answers on its own.
- **Later rounds:** every agent sees its own previous answer, plus the other agents' previous answers and reasoning
  (the last `show_chars` characters, default 1500). Older rounds are shown as answers only. It may keep or change
  its answer.
- **End:** the group answer is the most common final answer. A tie counts as the share of the tied answers that
  are right.
- **N = 1** is the no-debate baseline: the same rounds, but the agent sees only its own previous answer.

Answers are given as `ANSWER: <number>` for GSM8K and `ANSWER: <letter>` for the multiple-choice sets.

| env_kwargs | default | |
|---|---|---|
| `dataset` | `gsm8k` | `gsm8k` (math, numeric answer), `mmlu_pro` (14 subjects, up to 10 options) or `supergpqa` (SuperGPQA 'hard': mostly science and engineering, up to 10 options) or `hle` (Humanity's Last Exam, text-only multiple choice: 5-26 options) |
| `num_rounds` | 3 | |
| `show_chars` | 1500 | how much of each other agent's reasoning is shown |

**Questions.** `testbed/data/<dataset>.jsonl` holds a fixed sample of 500 test questions per set, shuffled with
seed 0 (`scripts/make_debate_data.py`). Episode i plays question i in every run and at every N: the runner passes
the episode index to adapters with `uses_episode = True`. All conditions and group sizes therefore answer the same
questions.

**Rewards and scores.**
- **Reward:** 1 for a correct final answer, 0 otherwise. Earlier rounds give 0.
- **`game_scores`:** the same per player, so welfare is the number of correct final answers.
- **Every step's `info`:** each agent's answer, whether it was correct, `n_correct`, the majority answer (or the
  tied answers) and `majority_correct`. From these, you can count answer switches from right to wrong and from
  wrong to right.

**HLE** asks not to redistribute its questions, so `testbed/data/hle.jsonl` is not in git: build it on each machine from the pinned revision (`scripts/make_debate_data.py`, after access is granted on Hugging Face). Some HLE questions list their options in the question and only letters as choices; a few have options truncated in the source.

**Sweep.** `config/sweeps/debate_acc.yaml` covers N = 1–5 on all three sets, with 50 questions each. GSM8K turned out too easy for the 9B model (one agent: ~92% right). `config/sweeps/debate_hard_pilot.yaml` measures the unsteered SuperGPQA accuracy first (the aim is 30–50% for one agent).
