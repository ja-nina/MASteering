"""Renderer for Hanabi."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext

_COLORS = ["red", "yellow", "green", "blue", "white"]


class HanabiRenderer:
    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        return "\n".join([
            "You are playing Hanabi — a cooperative card game.",
            "",
            "RULES:",
            "- You can see all other players' cards but NOT your own.",
            "- On your turn, do ONE of:",
            "    PLAY <pos>                      — play card at position (1-indexed) onto the fireworks",
            "    DISCARD <pos>                   — discard card at position; gain a clue token",
            "    HINT <player_id> COLOR <color>  — tell a player which of their cards match a color",
            "    HINT <player_id> RANK <rank>    — tell a player which of their cards match a rank (1–5)",
            "- Clue tokens: start at 8; spend 1 per hint, gain 1 per discard (max 8).",
            "- Fuse tokens: start at 3; lose 1 per wrong play. Reach 0 → game over.",
            "- Score = number of cards successfully played (max 25).",
            "- A color must be played in strict order: 1, 2, 3, 4, 5.",
            "- Goal: maximize the final score cooperatively.",
            "",
            "Respond with exactly one action in the format shown above.",
        ])

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs     = raw_obs
        is_turn = obs.get("is_my_turn", False)
        lines   = [
            f"=== Hanabi | Score: {obs.get('score', 0)}/25 | "
            f"Clues: {obs.get('clue_tokens', '?')}/8 | "
            f"Fuses: {obs.get('fuse_tokens', '?')}/3 | "
            f"Deck: {obs.get('deck_remaining', '?')} cards left ===",
            "",
        ]

        # Fireworks piles
        fw = obs.get("fireworks", {})
        fw_str = "  ".join(
            f"{c.upper()}: {fw.get(c, 0)}" for c in _COLORS
        )
        lines += [f"Fireworks: {fw_str}", ""]

        # Other players' hands (fully visible)
        for pid, hand in obs.get("others_hands", {}).items():
            cards_str = "  ".join(
                f"[{i+1}] {c['color'][0].upper()}{c['rank']}"
                for i, c in enumerate(hand)
            )
            lines.append(f"{pid}'s hand: {cards_str}")
        lines.append("")

        # Own hand with accumulated clues
        own_clues = obs.get("own_clues", [])
        own_parts = []
        for i, clue in enumerate(own_clues):
            colors_known = ", ".join(sorted(clue.get("colors", [])))
            ranks_known  = ", ".join(str(r) for r in sorted(clue.get("ranks", [])))
            hint_str = ""
            if colors_known:
                hint_str += f"color={colors_known} "
            if ranks_known:
                hint_str += f"rank={ranks_known}"
            own_parts.append(f"[{i+1}] {hint_str.strip() or '??'}")
        lines.append(f"Your hand: {' | '.join(own_parts)}")
        lines.append("(You cannot see your own cards — the above shows accumulated hints only.)")
        lines.append("")

        # Recent action history
        if context.history:
            lines.append("Recent actions:")
            for entry in context.history[-8:]:
                actor = entry.get("player", "?")
                act   = entry.get("action", "?")
                if act == "play":
                    c   = entry.get("card", {})
                    tag = f"{c.get('color','?')[0].upper()}{c.get('rank','?')}"
                    res = entry.get("result", "?")
                    desc = f"played {tag} → {res}"
                elif act == "discard":
                    c   = entry.get("card", {})
                    tag = f"{c.get('color','?')[0].upper()}{c.get('rank','?')}"
                    desc = f"discarded {tag}"
                elif act == "hint":
                    tgt  = entry.get("target", "?")
                    htyp = entry.get("hint_type", "?")
                    val  = entry.get("value", "?")
                    desc = f"hinted {tgt}: {htyp.upper()}={val}"
                else:
                    desc = act
                lines.append(f"  {actor}: {desc}")
            lines.append("")

        if is_turn:
            lines.append("It is YOUR turn. Choose one action:")
            lines.append("  PLAY <pos> | DISCARD <pos> | HINT <player_id> COLOR <color> | HINT <player_id> RANK <rank>")
        else:
            lines.append(f"It is {obs.get('current_player', '?')}'s turn. (You are observing.)")

        return "\n".join(lines)
