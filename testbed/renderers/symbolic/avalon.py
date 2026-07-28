"""Renderer for The Resistance: Avalon."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext

_ROLE_FLAVOR: dict[str, str] = {
    "Merlin":         "You are Merlin. You serve the forces of Good and can see most evil players.",
    "Percival":       "You are Percival. You serve Good and can see two players — one is Merlin, one may be Morgana.",
    "Loyal Servant":  "You are a Loyal Servant of Arthur. You serve the forces of Good.",
    "Mordred":        "You are Mordred. You serve the forces of Evil and are hidden from Merlin.",
    "Morgana":        "You are Morgana. You serve Evil and appear as Merlin to Percival.",
    "Oberon":         "You are Oberon. You serve Evil but are unknown to the other evil players.",
    "Spy":            "You are a Spy serving the forces of Evil.",
}

_PHASE_INSTRUCTIONS: dict[str, str] = {
    "propose": (
        "You are the current Leader.  Nominate {team_size} player(s) for Quest {quest_num}.\n"
        "Respond with: TEAM: player_X, player_Y  (comma-separated agent IDs)"
    ),
    "team_vote": (
        "Vote whether to APPROVE or REJECT the proposed team for Quest {quest_num}.\n"
        "Proposed team: {proposed_team}.\n"
        "Respond with: VOTE: APPROVE  or  VOTE: REJECT"
    ),
    "quest_vote": (
        "You are on the quest team.  Choose whether to make the quest SUCCEED or FAIL.\n"
        "Evil players may choose FAIL; Good players must choose SUCCESS.\n"
        "Respond with: QUEST: SUCCESS  or  QUEST: FAIL"
    ),
    "assassinate": (
        "Good has won 3 quests.  As the Assassin, identify Merlin to win the game for Evil.\n"
        "Living players: {living}.\n"
        "Respond with: ASSASSINATE: player_X"
    ),
}


class AvalonRenderer:
    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        obs  = raw_obs or {}
        role = obs.get("role", "Unknown")
        side = obs.get("side", "unknown")
        role_desc = _ROLE_FLAVOR.get(role, f"You are playing as {role}.")

        lines = [
            "You are playing The Resistance: Avalon.",
            "",
            role_desc,
            "",
            "GAME RULES:",
            "- 5 quests will be attempted. Good wins 3 successes; Evil needs 3 failures.",
            "- Each round: the Leader proposes a team → everyone votes → if approved the team secretly plays Success/Fail.",
            "- 5 consecutive rejections → Evil wins automatically.",
            "- If Good wins 3 quests: Evil may still win by correctly assassinating Merlin.",
            "",
            f"Your side: {side.upper()}.",
        ]

        # Role-specific knowledge
        known_evil = obs.get("known_evil", [])
        if known_evil:
            lines += ["", f"You know these players are evil: {', '.join(known_evil)}."]

        known_merlin = obs.get("known_merlin_candidates", [])
        if known_merlin:
            lines += ["", f"One of these players is Merlin (the other may be Morgana): {', '.join(known_merlin)}."]

        return "\n".join(lines)

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs = raw_obs
        lines = [
            f"=== Avalon — Quest {obs.get('quest_num', '?')} of 5 ===",
            f"Quests won by Good: {obs.get('quest_wins', 0)}  |  Quests failed: {obs.get('quest_losses', 0)}",
            f"Reject streak (auto-evil-win at 5): {obs.get('reject_streak', 0)}",
            f"Living players: {', '.join(obs.get('living', []))}",
            f"Current Leader: {obs.get('leader', '?')}",
            "",
        ]

        # Quest history
        for entry in obs.get("quest_history", []):
            result = "✓ SUCCEEDED" if entry.get("succeeded") else "✗ FAILED"
            team   = ", ".join(entry.get("team", []))
            lines.append(f"Quest {entry.get('quest', '?')}: {result} (team: {team})")
        if obs.get("quest_history"):
            lines.append("")

        # Phase-specific instruction
        phase = obs.get("phase", "")
        instr = _PHASE_INSTRUCTIONS.get(phase, "")
        if instr:
            lines.append(instr.format(
                team_size     = obs.get("team_size", "?"),
                quest_num     = obs.get("quest_num", "?"),
                proposed_team = ", ".join(obs.get("proposed_team", [])),
                living        = ", ".join(obs.get("living", [])),
            ))

        return "\n".join(lines)
