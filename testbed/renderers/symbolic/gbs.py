from __future__ import annotations

from testbed.types import RawObs, RenderContext


class GBSRenderer:
    def system_prompt(self, agent_id: str) -> str:
        return (
            f"You are {agent_id} in a cooperative group binary search game. "
            "There is a hidden target integer. Each round all players guess. "
            "After each round you learn the group's median guess and whether the "
            "target is higher or lower than that median. Work with the group to "
            "converge on the target."
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
