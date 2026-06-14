from __future__ import annotations

from testbed.types import RawObs, RenderContext


class BeautyContestRenderer:
    def system_prompt(self, agent_id: str) -> str:
        return (
            f"You are {agent_id} in a multi-player Keynesian beauty contest. "
            "Each round, every player picks an integer. The winning number is "
            "2/3 of the average of all picks. The player closest to that winning "
            "number wins the round. Reason about what others will pick, then "
            "respond with your chosen integer."
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        low, high = raw_obs["low"], raw_obs["high"]
        rnd = raw_obs["round_index"] + 1
        lines = [
            f"Round {rnd}. There are {raw_obs['num_players']} players.",
            f"Choose an integer between {low} and {high} (inclusive).",
        ]
        if raw_obs["history"]:
            last = raw_obs["history"][-1]
            lines.append(
                f"Last round the average was {last['mean']:.2f} and the winning "
                f"target (2/3 of average) was {last['target']:.2f}."
            )
        lines.append("Respond with your integer choice in the form: CHOICE: <number>")
        return "\n".join(lines)
