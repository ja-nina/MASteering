from __future__ import annotations

from testbed.types import RawObs, RenderContext

_FEEDBACK_DESC = {
    "exact": (
        "After each round you learn the group sum and the exact signed error — "
        "e.g. 'too HIGH by 23' or 'too LOW by 7'. "
        "A positive error means the sum overshot the target; negative means it fell short."
    ),
    "directional": (
        "After each round you learn the group sum and the direction only — "
        "'too HIGH' or 'too LOW' — but not by how much. "
        "You must estimate the magnitude from the group sum and adjust accordingly."
    ),
}


class GBSRenderer:
    def system_prompt(self, agent_id: str) -> str:
        # feedback description is injected at render time because we don't
        # know the feedback mode until we have an observation; use the exact
        # description as a safe default for the static system prompt slot.
        # The actual mode is described again in the first user prompt.
        return (
            f"You are {agent_id} in a cooperative Group Sum game.\n\n"
            "RULES\n"
            "- There is a hidden, unchanging, target number. Your group must make your individual "
            "contributions sum exactly to that target.\n"
            "- Every round, each player simultaneously submits a non-negative integer.\n"
            "- After each round you receive feedback on how close the group sum was "
            "to the target (the exact feedback mode is described below).\n"
            "- You do NOT see other players' individual submissions — only the group total.\n"
            "- The game ends when the group sum equals the target, or after the maximum "
            "number of rounds.\n\n"
            "Respond only in the form: CONTRIBUTION: <integer>"
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        rnd = raw_obs["round_index"] + 1
        n = raw_obs["num_players"]
        feedback_mode = raw_obs.get("feedback", "exact")

        lines = [
            f"Round {rnd}. There are {n} players.",
            f"Feedback mode: {_FEEDBACK_DESC[feedback_mode]}",
        ]

        if raw_obs["history"]:
            lines.append("Round history:")
            for h in raw_obs["history"]:
                my_contrib = h["contributions"].get(agent_id, "?")
                error = h["error"]
                if h["direction"] == "correct":
                    result = "CORRECT — target reached!"
                elif feedback_mode == "exact":
                    result = f"too HIGH by {error}" if error > 0 else f"too LOW by {abs(error)}"
                else:
                    result = "too HIGH" if error > 0 else "too LOW"
                lines.append(
                    f"  Round {h['round']}: group sum={h['group_sum']} "
                    f"({result}) | your contribution={my_contrib}"
                )

        lines.append("Respond with your contribution in the form: CONTRIBUTION: <integer>")
        return "\n".join(lines)
