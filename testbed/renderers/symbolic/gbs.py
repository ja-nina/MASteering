from __future__ import annotations

from testbed.types import RawObs, RenderContext
from testbed.visual import BLUE, GREEN, RED, bar_chart

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

# Verbatim from Riedl (2025, arXiv 2510.05174), plain condition.
_PICKING_RULES = (
    "You are playing a sum guessing game. "
    "Your goal is to help your group sum to the mystery number.\n"
    "Your guess range is 0 to 50.\n"
    "Always start with the efficient strategy in guessing games which is to use a "
    "binary search approach: guessing the midpoint of the current range.\n"
    "Always anchor your guess on the group feedback from previous rounds (too HIGH / too LOW).\n"
    "End your answer with: FINAL GUESS: [0-50]"
)


class GBSRenderer:
    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        # ── Picking variant (Riedl 2025) ─────────────────────────────────────
        if (raw_obs or {}).get("hide_group_size"):
            return _PICKING_RULES

        # ── Standard GBS ────────────────────────────────────────────────────
        return (
            f"You are {agent_id} in a cooperative Group Sum game.\n\n"
            "RULES\n"
            "- There is a hidden, unchanging, target number. Your group must make your individual "
            "numbers sum exactly to that target.\n"
            "- Every round, each player simultaneously submits a non-negative integer.\n"
            "- After each round players receive feedback on how close the group sum was "
            "to the target (the exact feedback mode is described below).\n"
            "- Players do NOT see other players' individual submissions — only the group total.\n"
            "- The game ends when the group sum equals the target, or after the maximum number "
            "of rounds. The current round number and rounds remaining are shown each turn.\n\n"
            "Respond in the form: NUMBER: <integer>"
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        n = raw_obs["num_players"]
        feedback_mode = raw_obs.get("feedback", "exact")
        hide_group_size = raw_obs.get("hide_group_size", False)
        round_str = self._round(raw_obs)

        # ── Picking variant — compact paper format ──────────────────────────
        if hide_group_size:
            lines = [round_str, "", "Game History:"]
            if raw_obs["history"]:
                for h in raw_obs["history"]:
                    my_contrib = h["contributions"].get(agent_id, "?")
                    if h["direction"] == "correct":
                        result = "CORRECT — target reached!"
                    elif h["direction"] == "too_high":
                        result = "too HIGH"
                    else:
                        result = "too LOW"
                    lines.append(f"Round {h['round']}: Your guess: {my_contrib} / Result: {result}")
            else:
                lines.append("No guesses yet.")
            lines.append("")
            lines.append("What is your guess this round? End your answer with: FINAL GUESS: [0-50]")
            return "\n".join(lines)

        # ── Standard GBS format ──────────────────────────────────────────────
        lines = [
            f"{round_str} There are {n} players.",
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
                    f"({result}) | your number={my_contrib}"
                )

        lines.append("Respond in the form: NUMBER: <integer>")
        return "\n".join(lines)

    modes = ("text", "image", "both")

    @staticmethod
    def _round(raw_obs) -> str:
        rnd, num_rounds = raw_obs["round_index"] + 1, raw_obs.get("num_rounds")
        if num_rounds is None:
            return f"Round {rnd}."
        left = num_rounds - rnd
        return f"Round {rnd} of {num_rounds} ({left} round{'s' if left != 1 else ''} remaining after this)."

    def render_visual(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        if raw_obs.get("hide_group_size"):
            return "\n".join([self._round(raw_obs), "",
                              "Your past guesses and the group result of each round are shown in the image.", "",
                              "What is your guess this round? End your answer with: FINAL GUESS: [0-50]"])
        return "\n".join([f"{self._round(raw_obs)} There are {raw_obs['num_players']} players.",
                          f"Feedback mode: {_FEEDBACK_DESC[raw_obs.get('feedback', 'exact')]}",
                          "The group sum of every past round, its feedback and your own number are shown in the image.",
                          "Respond in the form: NUMBER: <integer>"])

    def render_images(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> list:
        hist, picking = raw_obs["history"], raw_obs.get("hide_group_size")
        exact = raw_obs.get("feedback", "exact") == "exact" and not picking

        def mark(h):
            if h["direction"] == "correct":
                return "HIT"
            if exact:
                return f"{h['error']:+d}"
            return "HIGH" if h["direction"] == "too_high" else "LOW"

        colors = [GREEN if h["direction"] == "correct" else RED if h["direction"] == "too_high" else BLUE
                  for h in hist]
        if picking:  # the picking variant never reveals the group sum: only own guesses
            return [bar_chart([h["contributions"].get(agent_id) for h in hist], [f"R{h['round']}" for h in hist],
                              "your guesses (red = group too high, blue = too low)", colors=colors,
                              marks=[mark(h) for h in hist])]
        return [bar_chart([h["group_sum"] for h in hist],
                          [f"R{h['round']} you:{h['contributions'].get(agent_id)}" for h in hist],
                          "group sum per round (red = too high, blue = too low)", colors=colors,
                          marks=[mark(h) for h in hist])]
