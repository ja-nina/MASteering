from __future__ import annotations

from testbed.types import RawObs, RenderContext


class GBSRenderer:
    def system_prompt(self, agent_id: str) -> str:
        return (
            f"You are {agent_id} in a cooperative Group Binary Search game.\n\n"
            "RULES\n"
            "- There is a hidden target integer somewhere in a range you will be told each round.\n"
            "- Every round, all players simultaneously submit a guess.\n"
            "- After each round you learn the group's median guess and whether the target is "
            "HIGHER or LOWER than that median. The search bounds narrow accordingly.\n"
            "- The game ends early when any player guesses the target exactly.\n\n"
            "Respond only in the form: GUESS: <integer>"
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        low, high = raw_obs["low"], raw_obs["high"]
        rnd = raw_obs["round_index"] + 1
        lines = [
            f"Round {rnd}. Hidden target is between {low} and {high} (inclusive).",
        ]
        if raw_obs["history"]:
            last = raw_obs["history"][-1]
            lines.append(
                f"Last round the group median was {last['median']} and the target "
                f"is {last['direction']} than that."
            )
        lines.append("Respond with your integer guess in the form: GUESS: <number>")
        return "\n".join(lines)
