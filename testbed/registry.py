"""Map a game spec to its (adapter, renderer, parser) triple.

To add a new symbolic game:
  1. Create testbed/envs/symbolic/<game_id>.py  (subclass SymbolicAdapter)
     testbed/renderers/symbolic/<game_id>.py    (implement TextRenderer protocol)
     testbed/parsers/symbolic/<game_id>.py      (implement ActionParser protocol)
  2. Import the three classes below and add them to _SYMBOLIC.
  3. Add a config under config/games/<game_id>/ and docs/games/<game_id>.md.
See docs/games/adding_a_new_game.md for the full checklist.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.envs.textarena.ta_adapter import TextArenaAdapter
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.parsers.symbolic.gbs import GBSParser
from testbed.parsers.textarena import TextArenaParser
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.renderers.symbolic.gbs import GBSRenderer
from testbed.renderers.textarena import TextArenaRenderer

# ── Symbolic games ─────────────────────────────────────────────────────────────
# Add new games here: "game_id": (AdapterCls, RendererCls, ParserCls)
# Uncomment each block once the adapter's submit() + _observation() are implemented.
_SYMBOLIC = {
    "beauty_contest": (BeautyContestAdapter, BeautyContestRenderer, BeautyContestParser),
    "gbs": (GBSAdapter, GBSRenderer, GBSParser),
    # Faithful replication of Riedl (2025, arXiv 2510.05174) — mechanically
    # identical to gbs, distinguished only by env_kwargs (hide_group_size=True,
    # feedback="directional", persona_mode, personas list).
    "gbs_exact_replication": (GBSAdapter, GBSRenderer, GBSParser),
    # ── Skeleton games (stub adapters — uncomment after implementing submit()) ──
    # "avalon": (AvalonAdapter, AvalonRenderer, AvalonParser),
    # "hanabi": (HanabiAdapter, HanabiRenderer, HanabiParser),
    # "werewolf": (WerewolfAdapter, WerewolfRenderer, WerewolfParser),
    # "overcooked": (OvercookedAdapter, OvercookedRenderer, OvercookedParser),
}

# ── Lazy imports for skeleton games ───────────────────────────────────────────
# Kept separate so import errors in unfinished games don't break the main registry.
# Swap the commented block in _SYMBOLIC above once a game is implemented.
def _load_avalon():
    from testbed.envs.symbolic.avalon import AvalonAdapter
    from testbed.renderers.symbolic.avalon import AvalonRenderer
    from testbed.parsers.symbolic.avalon import AvalonParser
    return AvalonAdapter, AvalonRenderer, AvalonParser

def _load_hanabi():
    from testbed.envs.symbolic.hanabi import HanabiAdapter
    from testbed.renderers.symbolic.hanabi import HanabiRenderer
    from testbed.parsers.symbolic.hanabi import HanabiParser
    return HanabiAdapter, HanabiRenderer, HanabiParser

def _load_werewolf():
    from testbed.envs.symbolic.werewolf import WerewolfAdapter
    from testbed.renderers.symbolic.werewolf import WerewolfRenderer
    from testbed.parsers.symbolic.werewolf import WerewolfParser
    return WerewolfAdapter, WerewolfRenderer, WerewolfParser

def _load_overcooked():
    from testbed.envs.symbolic.overcooked import OvercookedAdapter
    from testbed.renderers.symbolic.overcooked import OvercookedRenderer
    from testbed.parsers.symbolic.overcooked import OvercookedParser
    return OvercookedAdapter, OvercookedRenderer, OvercookedParser

_SYMBOLIC_LAZY = {
    "avalon":     _load_avalon,
    "hanabi":     _load_hanabi,
    "werewolf":   _load_werewolf,
    "overcooked": _load_overcooked,
}


def build_game(*, family: str, game_id: str, num_players: int,
               env_kwargs: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    if family == "symbolic":
        if game_id in _SYMBOLIC:
            AdapterCls, RendererCls, ParserCls = _SYMBOLIC[game_id]
        elif game_id in _SYMBOLIC_LAZY:
            AdapterCls, RendererCls, ParserCls = _SYMBOLIC_LAZY[game_id]()
        else:
            raise ValueError(
                f"Unknown symbolic game: {game_id!r}. "
                f"Available: {sorted(_SYMBOLIC) + sorted(_SYMBOLIC_LAZY)}"
            )
        env = AdapterCls(num_players=num_players, **env_kwargs)
        return env, RendererCls(), ParserCls()
    if family == "textarena":
        env = TextArenaAdapter(env_id=game_id, num_players=num_players, **env_kwargs)
        return env, TextArenaRenderer(), TextArenaParser()
    raise ValueError(f"Unknown game family: {family}")
