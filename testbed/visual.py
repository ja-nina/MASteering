"""Image observations: observation modes and small PIL drawing helpers shared by renderers.

A renderer that can show its state as a picture implements, next to `render()`:
    render_images(raw_obs, agent_id, context) -> list of PIL images (the game state)
    render_visual(raw_obs, agent_id, context) -> str: the prompt without what the images show
and lists its supported modes in `modes`. A run's `observation` picks one:
    text   render()                                  (default; every renderer)
    image  render_visual() + render_images()         state only in the picture
    both   render()        + render_images()
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

OBSERVATION_MODES = ("text", "image", "both")

BG = (245, 242, 235)
INK = (30, 30, 30)
GREY = (150, 150, 150)
LIGHT = (215, 210, 200)
RED = (205, 55, 50)
GREEN = (60, 150, 70)
BLUE = (55, 95, 190)
YELLOW = (230, 190, 40)
WHITE = (255, 255, 255)
CARD_COLORS = {"red": RED, "yellow": YELLOW, "green": GREEN, "blue": BLUE, "white": (235, 235, 235)}
PLAYER_COLORS = [(205, 55, 50), (55, 95, 190), (60, 150, 70), (220, 140, 30), (140, 70, 170),
                 (40, 160, 170), (190, 90, 140), (110, 110, 110), (150, 120, 60), (30, 30, 30)]


def modes(renderer) -> Tuple[str, ...]:
    return tuple(getattr(renderer, "modes", ("text",)))


def observe(renderer, mode: str, raw_obs, agent_id, context) -> Tuple[str, list]:
    """(user prompt, images) for one agent's turn under an observation mode."""
    if mode == "text":
        return renderer.render(raw_obs, agent_id, context), []
    images = renderer.render_images(raw_obs, agent_id, context)
    text = renderer.render(raw_obs, agent_id, context) if mode == "both" else \
        renderer.render_visual(raw_obs, agent_id, context)
    return text, images


def fit(image, max_side: int):
    """Scale so the longer side is max_side (nearest-neighbour keeps pixel art crisp)."""
    from PIL import Image
    scale = max_side / max(image.size)
    if abs(scale - 1) < 1e-3:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.NEAREST if scale > 1 else Image.LANCZOS)


def font(size: int = 16):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: bitmap font only
        return ImageFont.load_default()


def canvas(width: int, height: int, color=BG):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), color)
    return img, ImageDraw.Draw(img)


def text_center(draw, xy: Tuple[float, float], text: str, size: int = 16, fill=INK) -> None:
    draw.text(xy, text, font=font(size), fill=fill, anchor="mm")


def title(draw, width: int, text: str, size: int = 18) -> None:
    """A header strip; the font shrinks until the text fits the width."""
    while size > 9 and draw.textlength(text, font=font(size)) > width - 10:
        size -= 1
    draw.rectangle([0, 0, width, 30], fill=LIGHT)
    text_center(draw, (width / 2, 15), text, size)


def grid_board(rows: int, cols: int, cell: int, cells, labels: bool = True, header: Optional[str] = None,
               row_labels: Optional[Sequence[str]] = None, col_labels: Optional[Sequence[str]] = None):
    """A labelled grid. `cells(draw, r, c, box)` paints one cell; box = (x0, y0, x1, y1)."""
    top = 34 if header else 0
    margin = 26 if labels else 4
    img, draw = canvas(margin + cols * cell + 4, top + margin + rows * cell + 4)
    if header:
        title(draw, img.width, header)
    for r in range(rows):
        for c in range(cols):
            box = (margin + c * cell, top + margin + r * cell, margin + (c + 1) * cell, top + margin + (r + 1) * cell)
            draw.rectangle(box, outline=GREY)
            cells(draw, r, c, box)
    if labels:
        for c in range(cols):
            text_center(draw, (margin + c * cell + cell / 2, top + margin / 2),
                        col_labels[c] if col_labels else str(c), 14)
        for r in range(rows):
            text_center(draw, (margin / 2, top + margin + r * cell + cell / 2),
                        row_labels[r] if row_labels else str(r), 14)
    return img


def _number(v) -> str:
    """A bar's value label; an integer too large for a float (an agent can answer with hundreds of digits) in
    scientific notation instead of crashing the render."""
    try:
        return f"{v:g}"
    except OverflowError:
        digits = str(abs(int(v)))
        return f"{'-' if v < 0 else ''}{digits[0]}.{digits[1:3]}e+{len(digits) - 1}"


def _ratio(v, vmax) -> float:
    try:
        return v / vmax
    except OverflowError:  # a huge negative against a small maximum: far below the axis (clipped there)
        return -1e9 if v < 0 else 1e9


def bar_chart(values: List[Optional[float]], labels: List[str], header: str, width: int = 480,
              height: int = 300, colors: Optional[List[tuple]] = None, marks: Optional[List[str]] = None,
              hline: Optional[float] = None):
    """Vertical bars with value labels; `marks` are short strings drawn above each bar."""
    img, draw = canvas(width, height)
    title(draw, width, header)
    if not values:
        text_center(draw, (width / 2, height / 2), "no rounds played yet", 16, GREY)
        return img
    top, bottom, left = 70, height - 30, 20
    vmax = max([v for v in values if v is not None] + ([hline] if hline is not None else []) + [1])
    slot = (width - 2 * left) / len(values)
    for i, v in enumerate(values):
        x0, x1 = left + i * slot + slot * 0.15, left + (i + 1) * slot - slot * 0.15
        if v is not None:
            y = min(bottom - (bottom - top) * _ratio(v, vmax), height - 18)  # a negative value hangs below the axis
            draw.rectangle([x0, min(y, bottom), x1, max(y, bottom)], fill=(colors[i] if colors else BLUE))
            text_center(draw, ((x0 + x1) / 2, y - 9), _number(v), 13)
            if marks and marks[i]:
                text_center(draw, ((x0 + x1) / 2, y - 24), marks[i], 13, RED)
        text_center(draw, ((x0 + x1) / 2, bottom + 12), labels[i], 13)
    if hline is not None:
        y = bottom - (bottom - top) * hline / vmax
        draw.line([left, y, width - left, y], fill=RED, width=2)
    draw.line([left, bottom, width - left, bottom], fill=INK)
    return img


def ring_of_players(names: List[str], header: str, size: int = 520, style=None):
    """Players on a circle. `style(name) -> (fill, outline, label_below)`."""
    import math
    img, draw = canvas(size, size + 40)
    title(draw, size, header)
    cx, cy, radius, r = size / 2, size / 2 + 30, size / 2 - 70, 30
    for i, name in enumerate(names):
        a = 2 * math.pi * i / max(1, len(names)) - math.pi / 2
        x, y = cx + radius * math.cos(a), cy + radius * math.sin(a)
        fill, outline, below = style(name) if style else (LIGHT, INK, "")
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=4)
        text_center(draw, (x, y), name[:8], 13)
        if below:
            text_center(draw, (x, y + r + 12), below, 12, INK)
    return img
