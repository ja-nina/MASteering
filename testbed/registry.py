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

from testbed.envs.symbolic.avalon import AvalonAdapter
from testbed.envs.symbolic.hanabi import HanabiAdapter
from testbed.envs.symbolic.overcooked import OvercookedAdapter
from testbed.envs.symbolic.werewolf import WerewolfAdapter
from testbed.envs.textarena.ta_adapter import TextArenaAdapter
from testbed.parsers.symbolic.avalon import AvalonParser
from testbed.parsers.symbolic.hanabi import HanabiParser
from testbed.parsers.symbolic.overcooked import OvercookedParser
from testbed.parsers.symbolic.werewolf import WerewolfParser
from testbed.parsers.textarena import TextArenaParser
from testbed.renderers.symbolic.avalon import AvalonRenderer
from testbed.renderers.symbolic.hanabi import HanabiRenderer
from testbed.renderers.symbolic.overcooked import OvercookedRenderer
from testbed.renderers.symbolic.werewolf import WerewolfRenderer
from testbed.renderers.textarena import TextArenaRenderer

_SYMBOLIC = {
    "avalon":    (AvalonAdapter,    AvalonRenderer,    AvalonParser),
    "hanabi":    (HanabiAdapter,    HanabiRenderer,    HanabiParser),
    "werewolf":  (WerewolfAdapter,  WerewolfRenderer,  WerewolfParser),
    "overcooked": (OvercookedAdapter, OvercookedRenderer, OvercookedParser),
}


def build_game(*, family: str, game_id: str, num_players: int,
               env_kwargs: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    # num_players may appear in env_kwargs when it duplicates agents.count in the
    # YAML; strip it so we don't pass it twice to the adapter constructor.
    env_kwargs = {k: v for k, v in env_kwargs.items() if k != "num_players"}

    if family == "symbolic":
        if game_id not in _SYMBOLIC:
            raise ValueError(
                f"Unknown symbolic game: {game_id!r}. "
                f"Available: {sorted(_SYMBOLIC)}"
            )
        AdapterCls, RendererCls, ParserCls = _SYMBOLIC[game_id]
        env = AdapterCls(num_players=num_players, **env_kwargs)
        return env, RendererCls(), ParserCls()

    if family == "textarena":
        env = TextArenaAdapter(env_id=game_id, num_players=num_players, **env_kwargs)
        return env, TextArenaRenderer(), TextArenaParser()

    raise ValueError(f"Unknown game family: {family!r}")
