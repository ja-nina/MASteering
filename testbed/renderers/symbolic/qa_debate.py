from __future__ import annotations

from testbed.types import RawObs, RenderContext


def _fmt(answer) -> str:
    return "no answer" if answer is None else str(answer)


class QADebateRenderer:
    modes = ("text",)

    def system_prompt(self, agent_id: str, raw_obs=None) -> str:
        n = raw_obs["num_players"] if raw_obs else "several"
        rounds = raw_obs["num_rounds"] if raw_obs else "a few"
        form = "ANSWER: <letter>" if raw_obs and raw_obs["dataset"] == "mmlu_pro" else "ANSWER: <number>"
        if n == 1:  # the no-debate baseline: the same rounds, alone
            return (f"You are {agent_id}, answering a question over {rounds} rounds. After each round you see your "
                    "previous answer and may keep or change it. You are scored on whether your final answer is "
                    f"correct.\n\nReason step by step, then end your reply with: {form}")
        return (
            f"You are {agent_id}, one of {n} agents answering the same question together over {rounds} rounds.\n\n"
            "RULES\n"
            "- Round 1: answer the question on your own.\n"
            "- Later rounds: you see the other agents' previous answers and reasoning. Use them to reconsider; "
            "you may keep or change your answer.\n"
            "- After the last round, the group's answer is the most common final answer. Each agent is scored on "
            "whether its own final answer is correct.\n\n"
            f"Reason step by step, then end your reply with: {form}"
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        ds, rnd = raw_obs["dataset"], raw_obs["round_index"] + 1
        lines = [f"Round {rnd} of {raw_obs['num_rounds']}.", "", "QUESTION", raw_obs["question"]]
        if raw_obs["options"]:
            lines += [""] + [f"{'ABCDEFGHIJ'[i]}. {o}" for i, o in enumerate(raw_obs["options"])]
        hist = raw_obs["history"]
        if hist:
            last, k = hist[-1], raw_obs["show_chars"]
            if len(hist) > 1:
                lines += ["", "EARLIER ROUNDS (answers only)"]
                for h in hist[:-1]:
                    lines.append(f"  Round {h['round']}: " + ", ".join(
                        f"{'you' if p == agent_id else p}: {_fmt(a)}" for p, a in h["answers"].items()))
            lines += ["", f"YOUR ANSWER IN ROUND {last['round']}: {_fmt(last['answers'].get(agent_id))}"]
            if raw_obs["num_players"] == 1:  # alone (the no-debate baseline): reconsider its own answer
                lines += ["", "Reconsider your answer."]
            else:
                lines += ["", f"THE OTHER AGENTS IN ROUND {last['round']}"]
                for p, a in last["answers"].items():
                    if p == agent_id:
                        continue
                    t = last["texts"].get(p, "")
                    t = t if len(t) <= k else "..." + t[-k:]
                    lines += [f"--- {p} (answer: {_fmt(a)}) ---", t or "(no reasoning given)"]
                lines += ["", "Reconsider the question in light of the other agents' answers."]
        form = "ANSWER: <letter>" if ds == "mmlu_pro" else "ANSWER: <number>"
        lines += ["", f"Reason step by step, then end your reply with: {form}"]
        return "\n".join(lines)
