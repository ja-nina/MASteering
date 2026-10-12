"""Renderer for Hanabi (text, or the table as an image)."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext
from testbed.visual import CARD_COLORS, GREY, INK, LIGHT, RED, canvas, text_center, title

_COLORS = ["red", "yellow", "green", "blue", "white"]


class HanabiRenderer:
    modes = ("text", "image", "both")

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
        obs = raw_obs
        lines = [
            f"=== Hanabi | Score: {obs.get('score', 0)}/25 | "
            f"Clues: {obs.get('clue_tokens', '?')}/8 | "
            f"Fuses: {obs.get('fuse_tokens', '?')}/3 | "
            f"Deck: {obs.get('deck_remaining', '?')} cards left ===",
            "",
        ]
        fw = obs.get("fireworks", {})
        lines += ["Fireworks: " + "  ".join(f"{c.upper()}: {fw.get(c, 0)}" for c in _COLORS), ""]
        for pid, hand in obs.get("others_hands", {}).items():
            cards = "  ".join(f"[{i + 1}] {c['color'][0].upper()}{c['rank']}" for i, c in enumerate(hand))
            lines.append(f"{pid}'s hand: {cards}")
        lines.append("")
        own = [f"[{i + 1}] {self._hint(clue) or '??'}" for i, clue in enumerate(obs.get("own_clues", []))]
        lines.append(f"Your hand: {' | '.join(own)}")
        lines.append("(You cannot see your own cards — the above shows accumulated hints only.)")
        lines.append("")
        return "\n".join(lines + self._tail(obs, context))

    def render_visual(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        return "\n".join(["The table is shown in the image: tokens, fireworks, the other players' cards, "
                          "and your own cards face down with the hints you have received.", ""]
                         + self._tail(raw_obs, context))

    def render_images(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> list:
        obs = raw_obs
        others = obs.get("others_hands", {})
        cw, ch, gap = 56, 78, 10
        width = max(560, 150 + (cw + gap) * max(5, obs.get("hand_size", 5)))
        img, draw = canvas(width, 170 + (len(others) + 1) * (ch + 34))
        title(draw, width, f"Hanabi   score {obs.get('score', 0)}/25   clues {obs.get('clue_tokens')}/8   "
                           f"fuses {obs.get('fuse_tokens')}/3   deck {obs.get('deck_remaining')}")
        text_center(draw, (60, 85), "Fireworks", 15)
        fw = obs.get("fireworks", {})
        for i, color in enumerate(_COLORS):
            self._card(draw, 130 + i * (cw + gap), 45, cw, ch, color, fw.get(color, 0))
        y = 45 + ch + 30
        for pid, hand in list(others.items()) + [(f"{agent_id} (you)", None)]:
            text_center(draw, (60, y + ch / 2), pid[:14], 14)
            if hand is not None:
                for i, card in enumerate(hand):
                    self._card(draw, 130 + i * (cw + gap), y, cw, ch, card["color"], card["rank"])
            else:
                for i, clue in enumerate(obs.get("own_clues", [])):
                    x = 130 + i * (cw + gap)
                    draw.rounded_rectangle([x, y, x + cw, y + ch], 6, fill=GREY, outline=INK, width=2)
                    text_center(draw, (x + cw / 2, y + ch / 2), "?", 30, LIGHT)
                    text_center(draw, (x + cw / 2, y + ch + 11), self._hint(clue, short=True) or "no hint", 11)
            for i in range(len(hand) if hand is not None else len(obs.get("own_clues", []))):
                text_center(draw, (130 + i * (cw + gap) + cw / 2, y - 9), f"[{i + 1}]", 11, RED)
            y += ch + 34
        return [img]

    @staticmethod
    def _card(draw, x, y, w, h, color, label) -> None:
        draw.rounded_rectangle([x, y, x + w, y + h], 6, fill=CARD_COLORS[color], outline=INK, width=2)
        text_center(draw, (x + w / 2, y + h / 2), str(label), 30, INK)

    @staticmethod
    def _hint(clue, short: bool = False) -> str:
        colors = ", ".join(sorted(clue.get("colors", [])))
        ranks = ", ".join(str(r) for r in sorted(clue.get("ranks", [])))
        if short:
            return " ".join(x for x in (colors, ranks) if x)
        return " ".join(x for x in (f"color={colors}" if colors else "", f"rank={ranks}" if ranks else "") if x)

    @staticmethod
    def _tail(obs, context: RenderContext) -> list:
        """Recent actions and the turn instruction (text in every observation mode)."""
        lines = []
        if context.history:
            lines.append("Recent actions:")
            for entry in context.history[-8:]:
                act, c = entry.get("action", "?"), entry.get("card", {})
                tag = f"{c.get('color', '?')[0].upper()}{c.get('rank', '?')}"
                if act == "play":
                    desc = f"played {tag} → {entry.get('result', '?')}"
                elif act == "discard":
                    desc = f"discarded {tag}"
                elif act == "hint":
                    desc = f"hinted {entry.get('target', '?')}: {entry.get('hint_type', '?').upper()}={entry.get('value', '?')}"
                else:
                    desc = act
                lines.append(f"  {entry.get('player', '?')}: {desc}")
            lines.append("")
        if obs.get("is_my_turn", False):
            lines.append("It is YOUR turn. Choose one action:")
            lines.append("  PLAY <pos> | DISCARD <pos> | HINT <player_id> COLOR <color> | HINT <player_id> RANK <rank>")
        else:
            lines.append(f"It is {obs.get('current_player', '?')}'s turn. (You are observing.)")
        return lines
