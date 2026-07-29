"""Renderer for The Resistance: Avalon.

System prompt design follows the framing used in LLM-Avalon research
(e.g., Shi et al. 2023 "Avalon's Game of Deception") — give each player
precise, unambiguous knowledge statements and a complete rules reference,
then a concise per-turn context.
"""
from __future__ import annotations

from testbed.types import RawObs, RenderContext

# ── game constants mirrored from the env ──────────────────────────────────────

_QUEST_SIZES: dict[int, list[int]] = {
    5:  [2, 3, 2, 3, 3],
    6:  [2, 3, 4, 3, 4],
    7:  [2, 3, 3, 4, 4],
    8:  [3, 4, 4, 5, 5],
    9:  [3, 4, 4, 5, 5],
    10: [3, 4, 4, 5, 5],
}

# Quest 4 at 7+ players needs 2 FAIL votes to fail
_QUEST4_NEEDS_2_FAILS: set[int] = {7, 8, 9, 10}

_EVIL_COUNTS: dict[int, int] = {5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4}


# ── role knowledge descriptions ───────────────────────────────────────────────

def _role_block(role: str, side: str, agent_id: str, known_evil: list[str],
                known_merlin: list[str], n_players: int) -> list[str]:
    """Return lines describing this agent's role and private knowledge.

    known_evil may include agent_id itself (the env lists all evil players);
    we strip self out here so the wording "other evil players" is accurate.
    """
    others_evil = [p for p in known_evil if p != agent_id]
    lines: list[str] = []

    if role == "Merlin":
        lines += [
            "ROLE: Merlin (Good)",
            "You serve Good and have special night-vision knowledge:",
            "  • You can see ALL evil players in the game, EXCEPT Mordred (who is hidden from Merlin).",
        ]
        if known_evil:
            lines += [
                f"  • These players are evil: {', '.join(known_evil)}",
                "    (If Mordred is in this game, they do NOT appear on this list.)",
            ]
        else:
            lines.append("  • No evil players are visible to you (all evil players may be hidden).")
        lines += [
            "Strategic note: Help Good succeed quests without revealing you are Merlin —",
            "if Good wins 3 quests, the Evil Assassin gets one shot to name you and win.",
        ]

    elif role == "Percival":
        lines += [
            "ROLE: Percival (Good)",
            "You serve Good and have special night-vision knowledge:",
            "  • You see exactly two players who appear to be Merlin — one IS Merlin (Good),",
            "    the other MAY be Morgana (Evil, appearing as Merlin).",
        ]
        if known_merlin:
            lines += [
                f"  • The two Merlin candidates you see: {', '.join(known_merlin)}",
                "    Deduce which is the real Merlin and protect them during the assassination phase.",
            ]

    elif role == "Loyal Servant":
        lines += [
            "ROLE: Loyal Servant of Arthur (Good)",
            "You serve Good and receive NO special night-vision knowledge.",
            "You have no information about who is evil — deduce it from proposals, votes, and quest outcomes.",
        ]

    elif role == "Mordred":
        lines += [
            "ROLE: Mordred (Evil)",
            "You serve Evil. Your special ability: Merlin CANNOT see you — you are invisible to Merlin's knowledge.",
            "You have night-vision knowledge of the other evil players (Oberon is hidden from the evil team):",
        ]
        if others_evil:
            lines.append(f"  • Your fellow evil players: {', '.join(others_evil)}")
        else:
            lines.append("  • You are the only evil player visible to the team (others may be Oberon).")
        lines.append("Coordinate with your evil allies to fail quests while exploiting your Merlin-invisibility.")

    elif role == "Morgana":
        lines += [
            "ROLE: Morgana (Evil)",
            "You serve Evil. Your special ability: Percival sees you as a possible Merlin — use this to deceive.",
            "You have night-vision knowledge of the other evil players (Oberon is hidden from the evil team):",
        ]
        if others_evil:
            lines.append(f"  • Your fellow evil players: {', '.join(others_evil)}")
        else:
            lines.append("  • You are the only evil player visible to the team (others may be Oberon).")

    elif role == "Oberon":
        lines += [
            "ROLE: Oberon (Evil)",
            "You serve Evil, but with two restrictions:",
            "  • You do NOT know who the other evil players are.",
            "  • The other evil players do NOT know you are evil.",
            "You must sabotage quests independently, without coordination.",
        ]

    elif role == "Spy":
        n_evil = _EVIL_COUNTS.get(n_players, 2)
        lines += [
            "ROLE: Spy (Evil)",
            f"You serve Evil. In this {n_players}-player game there are {n_evil} evil players total (including you).",
            "You have night-vision knowledge of the other evil players",
            "(Oberon, if present, is hidden from the evil team and does NOT appear below):",
        ]
        if others_evil:
            lines.append(f"  • Your fellow Spies: {', '.join(others_evil)}")
            lines.append("    You and your fellow Spies know each other. Coordinate to fail quests without detection.")
        else:
            lines.append("  • You appear to be the only evil player with shared knowledge (no other visible Spies).")

    else:
        lines.append(f"ROLE: {role} ({side.upper()})")

    return lines


# ── system prompt ─────────────────────────────────────────────────────────────

class AvalonRenderer:
    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        obs        = raw_obs or {}
        role       = obs.get("role", "Unknown")
        side       = obs.get("side", "unknown")
        known_evil = obs.get("known_evil", [])
        known_merlin = obs.get("known_merlin_candidates", [])
        n_players  = len(obs.get("living", [])) or 5
        n_evil     = _EVIL_COUNTS.get(n_players, 2)
        n_good     = n_players - n_evil
        sizes      = _QUEST_SIZES.get(n_players, [2, 3, 2, 3, 3])

        lines: list[str] = [
            "You are playing The Resistance: Avalon.",
            f"Your player ID: {agent_id}",
            "",
        ]

        # ── role + knowledge block ────────────────────────────────────────────
        lines += _role_block(role, side, agent_id, known_evil, known_merlin, n_players)
        lines.append("")

        # ── game overview ─────────────────────────────────────────────────────
        quest_str = "  ".join(f"Q{i+1}:{sz}" for i, sz in enumerate(sizes))
        lines += [
            "━━━━ GAME OVERVIEW ━━━━",
            f"Players: {n_players} total  ({n_good} Good, {n_evil} Evil)",
            f"Quest team sizes: {quest_str}",
        ]
        if n_players in _QUEST4_NEEDS_2_FAILS:
            lines.append("Special rule: Quest 4 requires 2 FAIL votes to fail (all others need only 1).")
        lines += [
            "",
            "WIN CONDITIONS:",
            "  GOOD wins by: (1) succeeding 3 of 5 quests, AND (2) the Assassin failing to identify Merlin.",
            "  EVIL wins by: (1) failing 3 quests, OR (2) forcing 5 consecutive proposal rejections,",
            "                OR (3) correctly assassinating Merlin after Good wins 3 quests.",
            "",
            "EACH QUEST ROUND proceeds in three phases:",
            "  1. PROPOSE  — The current Leader nominates exactly the required number of players.",
            "                 The Leader may include themselves.",
            "  2. VOTE     — ALL players simultaneously vote APPROVE or REJECT.",
            "                 If more than half approve → the team goes on the quest.",
            "                 If half or fewer approve → proposal is rejected, leadership passes left,",
            "                 and the reject streak increases. At 5 consecutive rejections, Evil wins.",
            "  3. QUEST    — Each team member secretly votes SUCCESS or FAIL.",
            "                 Good players MUST vote SUCCESS.",
            "                 Evil players may vote either SUCCESS or FAIL.",
            "                 The quest fails if the required number of FAIL votes are cast.",
            "                 After the quest, only the number of FAIL votes is revealed (not who voted what).",
        ]
        if role in ("Merlin", "Percival", "Loyal Servant"):
            lines += [
                "",
                "ASSASSINATION PHASE (if Good wins 3 quests and Merlin is in the game):",
                "  Evil's Assassin names one player as Merlin. If correct → Evil wins.",
                "  Merlin must be careful not to act too obviously — revealing themselves loses the game.",
            ]

        return "\n".join(lines)

    # ── per-turn render ───────────────────────────────────────────────────────

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs       = raw_obs
        phase     = obs.get("phase", "")
        living    = obs.get("living", [])
        leader    = obs.get("leader", "?")
        am_leader = (leader == agent_id)
        quest_num = obs.get("quest_num", 1)
        wins      = obs.get("quest_wins", 0)
        losses    = obs.get("quest_losses", 0)
        streak    = obs.get("reject_streak", 0)
        n_players = len(living) or 5
        sizes     = _QUEST_SIZES.get(n_players, [2, 3, 2, 3, 3])

        # ── header ────────────────────────────────────────────────────────────
        score_icons = []
        history     = obs.get("quest_history", [])
        for i in range(5):
            entry = next((e for e in history if e.get("quest") == i + 1), None)
            if entry is None:
                score_icons.append("[ ]")
            elif entry.get("succeeded"):
                score_icons.append("[✓]")
            else:
                score_icons.append("[✗]")

        lines = [
            f"=== Avalon | Quest {quest_num}/5 | Good: {wins} ✓  Evil: {losses} ✗ ===",
            f"Quest progress: {' '.join(score_icons)}",
            f"Proposal reject streak: {streak}/5  (5 = instant Evil win)",
            "",
            f"Players (you are {agent_id}): {', '.join(living)}",
            f"Current Leader: {leader}" + (" ← YOU" if am_leader else ""),
            "",
        ]

        # ── quest history with fail vote counts ───────────────────────────────
        if history:
            lines.append("Quest history:")
            for entry in history:
                q       = entry.get("quest", "?")
                team    = ", ".join(entry.get("team", []))
                fails   = entry.get("fail_votes", "?")
                result  = "SUCCEEDED" if entry.get("succeeded") else "FAILED"
                sz      = sizes[int(q) - 1] if isinstance(q, int) and q <= len(sizes) else "?"
                fail_needed = (2 if (isinstance(q, int) and q == 4
                                     and n_players in _QUEST4_NEEDS_2_FAILS) else 1)
                lines.append(
                    f"  Q{q} ({sz} players) — {result}  "
                    f"[team: {team} | FAIL votes cast: {fails}/{fail_needed} needed to fail]"
                )
            lines.append("")

        # ── phase-specific instruction ─────────────────────────────────────────
        if phase == "propose":
            team_size = obs.get("team_size", "?")
            others    = [p for p in living if p != agent_id]
            if am_leader:
                lines += [
                    f"PHASE: PROPOSE — You are the Leader for Quest {quest_num}.",
                    f"You must nominate exactly {team_size} player(s) for this quest.",
                    f"Choose from: {', '.join(living)}  (you may include yourself: {agent_id})",
                    "",
                    "Think about: which players have been on failed quests? who voted to approve suspicious teams?",
                    "",
                    f"Respond with exactly:  TEAM: player_a, player_b",
                    f"(provide exactly {team_size} player ID(s), comma-separated)",
                ]
            else:
                lines += [
                    f"PHASE: PROPOSE — {leader} is the Leader and is choosing a team.",
                    "(You are not the leader this round — wait for the vote phase.)",
                ]

        elif phase == "team_vote":
            team_size = obs.get("team_size", "?")
            proposed  = obs.get("proposed_team", [])
            lines += [
                f"PHASE: VOTE — {leader} proposes this team for Quest {quest_num}:",
                f"  Proposed team ({len(proposed)}/{team_size}): {', '.join(proposed)}",
                "",
                "All players vote simultaneously. If MORE than half approve → quest proceeds.",
                "If half or fewer approve → proposal rejected, reject streak increases, leadership passes left.",
                f"Current reject streak: {streak}/5  (reaching 5 = Evil wins immediately)",
                "",
                "Think about: do you trust the proposed team? are known-suspicious players included?",
                "",
                "Respond with exactly:  VOTE: APPROVE  or  VOTE: REJECT",
            ]

        elif phase == "quest_vote":
            proposed = obs.get("proposed_team", [])
            q_num    = quest_num - 1  # 0-indexed for fail_required lookup
            fail_needed = (2 if (quest_num == 4 and n_players in _QUEST4_NEEDS_2_FAILS) else 1)
            role     = obs.get("role", "")
            lines += [
                f"PHASE: QUEST — You are on the quest team for Quest {quest_num}.",
                f"Team members: {', '.join(proposed)}",
                f"This quest fails if {fail_needed} or more FAIL vote(s) are cast.",
                "Only the total number of FAIL votes will be revealed — not who voted what.",
                "",
            ]
            if role in ("Spy", "Morgana", "Mordred", "Oberon"):
                lines += [
                    "As an Evil player, you may vote FAIL to sabotage this quest.",
                    "Consider whether failing now risks exposing you.",
                ]
            else:
                lines += [
                    "As a Good player, you MUST vote SUCCESS.",
                ]
            lines += [
                "",
                "Respond with exactly:  QUEST: SUCCESS  or  QUEST: FAIL",
            ]

        elif phase == "assassinate":
            proposed = obs.get("proposed_team", [])
            lines += [
                "PHASE: ASSASSINATION — Good has won 3 quests.",
                "As the Assassin, you have one chance to identify Merlin.",
                "If you name the correct player, Evil wins. If wrong, Good wins.",
                "",
                f"Living players to choose from: {', '.join(living)}",
                "",
                "Recall: Merlin tried to guide Good to success without revealing themselves.",
                "Look for players who: voted consistently against suspicious teams,",
                "proposed well-balanced teams when leader, or seemed to 'know' who to avoid.",
                "",
                "Respond with exactly:  ASSASSINATE: player_id",
            ]

        return "\n".join(lines)
