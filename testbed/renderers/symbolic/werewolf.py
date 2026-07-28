"""Renderer for Werewolf."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext

_PHASE_HEADERS = {
    "night_werewolf": "NIGHT — Werewolves awaken",
    "night_seer":     "NIGHT — The Seer investigates",
    "night_doctor":   "NIGHT — The Doctor acts",
    "day_discussion": "DAY — Discussion",
    "day_vote":       "DAY — Vote to eliminate",
}

_PHASE_INSTRUCTIONS = {
    "night_werewolf": (
        "Choose one living player to eliminate tonight.\n"
        "Respond with: KILL: player_X"
    ),
    "night_seer": (
        "Choose one living player to investigate.  You will learn their true role.\n"
        "Respond with: INVESTIGATE: player_X"
    ),
    "night_doctor": (
        "Choose one living player to protect from elimination tonight.  "
        "You may choose yourself.\n"
        "Respond with: SAVE: player_X"
    ),
    "day_discussion": (
        "It is your turn to speak.  Share your observations, suspicions, or arguments. "
        "Be concise.\n"
        "Respond with: STATEMENT: <your message>"
    ),
    "day_vote": (
        "Vote to eliminate one player.  The player with the most votes is eliminated.  "
        "Ties are resolved randomly.\n"
        "Respond with: VOTE: player_X"
    ),
}


class WerewolfRenderer:
    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        obs  = raw_obs or {}
        role = obs.get("role", "Unknown")
        side = obs.get("side", "village")

        lines = [
            "You are playing Werewolf (social deduction game).",
            "",
            f"Your role: {role}.  Side: {side.upper()}.",
            "",
        ]

        if side == "werewolf":
            allies = obs.get("known_werewolves", [])
            if allies:
                lines += [f"Your werewolf allies: {', '.join(allies)}.", ""]
            lines += [
                "OBJECTIVE: Eliminate enough villagers so werewolves equal or outnumber them.",
                "  - Each night: choose a player to kill.",
                "  - During the day: avoid being voted out.",
            ]
        else:
            seer_k = obs.get("seer_knowledge", {})
            if seer_k:
                lines += [
                    "Roles you have already investigated:",
                    *[f"  {p}: {r}" for p, r in seer_k.items()],
                    "",
                ]
            lines += [
                "OBJECTIVE: Identify and vote out all werewolves.",
                "  - Use the day discussion and votes to eliminate suspects.",
                "  - Special roles: Seer investigates at night; Doctor can save one player.",
            ]

        lines += [
            "",
            "GAME FLOW: Night (werewolves kill, seer sees, doctor saves) → Day (discussion + vote). "
            "Repeat until one side wins.",
        ]
        return "\n".join(lines)

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs   = raw_obs
        phase = obs.get("phase", "")
        lines = [
            f"=== Werewolf — Day {obs.get('day_num', '?')} | "
            f"{_PHASE_HEADERS.get(phase, phase)} ===",
            "",
            f"Living players ({len(obs.get('living', []))}): {', '.join(obs.get('living', []))}",
            "",
        ]

        # Elimination log
        for entry in obs.get("elimination_log", []):
            role_reveal = f" (was {entry['role']})" if entry.get("role") else ""
            lines.append(
                f"  Day {entry.get('day', '?')}: {entry.get('eliminated', '?')}"
                f"{role_reveal} was eliminated by {entry.get('by', '?')}."
            )
        if obs.get("elimination_log"):
            lines.append("")

        # Recent statements from history
        day_statements = [h for h in context.history if h.get("phase") == "day_discussion"]
        for stmt in day_statements[-6:]:
            lines.append(f"  {stmt.get('player', '?')}: {stmt.get('statement', '')}")
        if day_statements:
            lines.append("")

        # Seer knowledge (only visible to seer)
        seer_k = obs.get("seer_knowledge", {})
        if seer_k:
            lines.append("Your investigation results:")
            for pid, role in seer_k.items():
                lines.append(f"  {pid}: {role}")
            lines.append("")

        # Phase instruction
        instr = _PHASE_INSTRUCTIONS.get(phase)
        if instr:
            lines.append(instr)

        return "\n".join(lines)
