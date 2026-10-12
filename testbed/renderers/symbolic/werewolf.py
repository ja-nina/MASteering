"""Renderer for Werewolf."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext
from testbed.visual import BLUE, GREEN, GREY, INK, LIGHT, RED, ring_of_players

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
        "Respond with: KILL: <name>"
    ),
    "night_seer": (
        "Choose one living player to investigate.  You will learn their true role.\n"
        "Respond with: INVESTIGATE: <name>"
    ),
    "night_doctor": (
        "Choose one living player to protect from elimination tonight.  "
        "You may choose yourself.\n"
        "Respond with: SAVE: <name>"
    ),
    "day_discussion": (
        "It is your turn to speak.  Share your observations, suspicions, or arguments. "
        "Be concise.\n"
        "Respond with: STATEMENT: <your message>"
    ),
    "day_vote": (
        "Vote to eliminate one player.  The player with the most votes is eliminated.  "
        "Ties are resolved randomly.\n"
        "Respond with: VOTE: <name>"
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

    modes = ("text", "image", "both")

    @staticmethod
    def _header(obs) -> str:
        phase = obs.get("phase", "")
        return f"=== Werewolf — Day {obs.get('day_num', '?')} | {_PHASE_HEADERS.get(phase, phase)} ==="

    def render_visual(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        """The village (who is alive, eliminations with revealed roles, what you know) is in the image."""
        lines = [self._header(raw_obs),
                 "The village is shown in the image: living and eliminated players (with revealed roles), "
                 "and the roles you know. You are outlined in blue.", ""]
        lines += self._discussion(context)
        instr = _PHASE_INSTRUCTIONS.get(raw_obs.get("phase", ""))
        return "\n".join(lines + ([instr] if instr else []))

    def render_images(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> list:
        obs = raw_obs
        dead = {e["eliminated"]: e for e in obs.get("elimination_log", []) if e.get("eliminated")}
        allies = set(obs.get("known_werewolves", []))
        known = dict(obs.get("seer_knowledge", {}))
        everyone = sorted(set(obs.get("living", [])) | set(dead))

        def style(name):
            outline = BLUE if name == agent_id else INK
            if name in dead:
                return GREY, outline, f"out day {dead[name].get('day', '?')}: {dead[name].get('role', '?')}"
            if name == agent_id:
                return LIGHT, outline, f"you: {obs.get('role')}"
            if name in allies:
                return RED, outline, "Werewolf (ally)"
            if name in known:
                return (RED if known[name] == "Werewolf" else GREEN), outline, f"seen: {known[name]}"
            return LIGHT, outline, ""

        return [ring_of_players(everyone, self._header(obs).strip("= "), style=style)]

    @staticmethod
    def _discussion(context: RenderContext) -> list:
        lines, current_day = [], None
        for stmt in (h for h in context.history if h.get("phase") == "discussion"):
            if stmt.get("day") != current_day:
                current_day = stmt.get("day")
                lines.append(f"  --- Day {current_day} discussion ---")
            lines.append(f"  {stmt.get('speaker', '?')}: {stmt.get('statement', '')}")
        return lines + ([""] if lines else [])

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs   = raw_obs
        phase = obs.get("phase", "")
        lines = [
            self._header(obs),
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

        lines += self._discussion(context)  # all statements, grouped by day

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
