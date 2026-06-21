from __future__ import annotations

from testbed.types import RawObs, RenderContext


class BeautyContestRenderer:
    def system_prompt(self, agent_id: str) -> str:
        return (
            f"You are {agent_id} playing a multi-player Keynesian beauty contest.\n\n"
            "RULES\n"
            "- Every round, each player simultaneously picks an integer from a given range.\n"
            "- The winning target = 2/3 × (average of all players' picks), rounded to one decimal.\n"
            "- The player whose guess is closest to the target wins that round. Ties are shared.\n"
            "- After each round you will see: the group average, the winning target, "
            "your own guess, and whether it was too high, too low, or a win.\n\n"
            "Respond in the form: CHOICE: <integer>"
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        low, high = raw_obs["low"], raw_obs["high"]
        rnd = raw_obs["round_index"] + 1
        lines = [
            f"Round {rnd}. There are {raw_obs['num_players']} players.",
            f"Choose an integer between {low} and {high} (inclusive).",
        ]
        if raw_obs["history"]:
            lines.append("Round history:")
            for i, h in enumerate(raw_obs["history"]):
                my_guess = h["choices"].get(agent_id)
                target = h["target"]
                won = agent_id in h["winners"]
                rnd_num = h.get("round", i + 1)
                summary = (
                    f"  Round {rnd_num}: group avg={h['mean']:.1f}, "
                    f"target={target:.1f}"
                )
                if my_guess is not None:
                    if won:
                        verdict = "WIN"
                    elif my_guess > target:
                        verdict = "too HIGH"
                    else:
                        verdict = "too LOW"
                    summary += f" | you guessed {my_guess:.0f} ({verdict})"
                lines.append(summary)
        lines.append("Respond in the form: CHOICE: <integer>")
        return "\n".join(lines)
