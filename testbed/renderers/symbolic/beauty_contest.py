from __future__ import annotations

from testbed.types import RawObs, RenderContext
from testbed.visual import BLUE, GREEN, RED, bar_chart


class BeautyContestRenderer:
    def system_prompt(self, agent_id: str, raw_obs=None) -> str:
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

    modes = ("text", "image", "both")

    def render_visual(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        return "\n".join([f"Round {raw_obs['round_index'] + 1}. There are {raw_obs['num_players']} players.",
                          f"Choose an integer between {raw_obs['low']} and {raw_obs['high']} (inclusive).",
                          "Your past guesses, each round's winning target and your result are shown in the image.",
                          "Respond in the form: CHOICE: <integer>"])

    def render_images(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> list:
        hist = raw_obs["history"]
        guesses = [h["choices"].get(agent_id) for h in hist]
        marks = ["WIN" if agent_id in h["winners"] else ("HIGH" if g is not None and g > h["target"] else "LOW")
                 for g, h in zip(guesses, hist)]
        colors = [GREEN if m == "WIN" else RED if m == "HIGH" else BLUE for m in marks]
        return [bar_chart(guesses, [f"R{h['round']} t={h['target']:.1f}" for h in hist],
                          "your guess per round (t = winning target = 2/3 of the average)",
                          colors=colors, marks=marks)]
