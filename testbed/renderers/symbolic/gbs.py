from __future__ import annotations

from testbed.types import RawObs, RenderContext


class GBSRenderer:
    def system_prompt(self, agent_id: str) -> str:
        return (
            f"You are {agent_id} in a cooperative Group Sum game.\n\n"
            "RULES\n"
            "- There is a hidden target number. Your group must make your individual "
            "contributions sum exactly to that target.\n"
            "- Every round, each player simultaneously submits a non-negative integer.\n"
            "- After each round you learn the group sum and the exact signed error: "
            "positive means the sum was too high, negative means too low.\n"
            "- You do NOT see other players' individual submissions — only the group total.\n"
            "- The game ends when the group sum equals the target, or after the maximum "
            "number of rounds.\n\n"
            "Respond only in the form: CONTRIBUTION: <integer>"
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        rnd = raw_obs["round_index"] + 1
        n = raw_obs["num_players"]
        lines = [f"Round {rnd}. There are {n} players."]

        if raw_obs["history"]:
            lines.append("Round history:")
            for h in raw_obs["history"]:
                my_contrib = h["contributions"].get(agent_id, "?")
                error = h["error"]
                if h["direction"] == "correct":
                    feedback = "CORRECT — target reached!"
                elif error > 0:
                    feedback = f"too HIGH by {error}"
                else:
                    feedback = f"too LOW by {abs(error)}"
                lines.append(
                    f"  Round {h['round']}: group sum={h['group_sum']} "
                    f"({feedback}) | your contribution={my_contrib}"
                )

        lines.append("Respond with your contribution in the form: CONTRIBUTION: <integer>")
        return "\n".join(lines)
