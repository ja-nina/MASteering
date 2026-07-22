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

# Verbatim from Riedl (2025, arXiv 2510.05174) — paper's three prompt conditions.
_PICKING_RULES = (
    "You are playing a sum guessing game. "
    "Your goal is to help your group sum to the mystery number.\n"
    "Your guess range is 0 to 50.\n"
    "Always start with the efficient strategy in guessing games which is to use a "
    "binary search approach: guessing the midpoint of the current range.\n"
    "Always anchor your guess on the group feedback from previous rounds (too HIGH / too LOW).\n"
    "End your answer with: FINAL GUESS: [0-50]"
)

_TOM_INSTRUCTION = (
    "\nOnly as a secondary approach, carefully think through step-by-step what "
    "others might guess and how the contributions of others contribute to the sum "
    "of the group guesses for the mystery number. "
    "Consider what roles other agents might be playing (e.g., guessing higher or lower) "
    "and adapt your own adjustment to complement the group."
)


class GBSRenderer:
    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        obs = raw_obs or {}
        persona_mode = obs.get("persona_mode", "plain")
        persona = obs.get("persona")

        # ── Picking / Persona variant (Riedl 2025) ──────────────────────────
        if obs.get("hide_group_size"):
            parts = []
            if persona and persona_mode in ("persona", "tom"):
                parts.append(persona)
                parts.append("")          # blank line between persona and rules
            parts.append(_PICKING_RULES)
            if persona_mode == "tom":
                parts.append(_TOM_INSTRUCTION)
            return "\n".join(parts)

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
        rnd = raw_obs["round_index"] + 1
        num_rounds = raw_obs.get("num_rounds")
        n = raw_obs["num_players"]
        feedback_mode = raw_obs.get("feedback", "exact")
        hide_group_size = raw_obs.get("hide_group_size", False)

        remaining = (num_rounds - rnd) if num_rounds is not None else None
        if remaining is not None:
            round_str = (
                f"Round {rnd} of {num_rounds} "
                f"({remaining} round{'s' if remaining != 1 else ''} remaining after this)."
            )
        else:
            round_str = f"Round {rnd}."

        # ── Picking / Persona variant — compact paper format ─────────────────
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
