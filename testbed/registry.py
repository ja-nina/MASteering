"""Map a game spec to its (adapter, renderer, parser) triple.

To add a new symbolic game:
  1. Create testbed/envs/symbolic/<game_id>.py  (subclass SymbolicAdapter)
     testbed/renderers/symbolic/<game_id>.py    (implement TextRenderer protocol)
     testbed/parsers/symbolic/<game_id>.py      (implement ActionParser protocol)
  2. Import the three classes below and add them to _SYMBOLIC.
  3. Add it to a sweep in config/sweeps/ and write docs/games/<game_id>.md.
See docs/games/adding_a_new_game.md for the full checklist.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

from testbed.envs.symbolic.avalon import AvalonAdapter
from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.envs.symbolic.board import BoardAdapter
from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.envs.symbolic.hanabi import HanabiAdapter
from testbed.envs.symbolic.harvest import HarvestAdapter
from testbed.envs.symbolic.overcooked import OvercookedAdapter
from testbed.envs.symbolic.werewolf import WerewolfAdapter
from testbed.envs.textarena.ta_adapter import TextArenaAdapter
from testbed.parsers.symbolic.avalon import AvalonParser
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.parsers.symbolic.board import BoardParser
from testbed.parsers.symbolic.gbs import GBSParser
from testbed.parsers.symbolic.hanabi import HanabiParser
from testbed.parsers.symbolic.harvest import HarvestParser
from testbed.parsers.symbolic.overcooked import OvercookedParser
from testbed.parsers.symbolic.werewolf import WerewolfParser
from testbed.parsers.textarena import TextArenaParser
from testbed.renderers.symbolic.avalon import AvalonRenderer
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.renderers.symbolic.board import BoardRenderer
from testbed.renderers.symbolic.gbs import GBSRenderer
from testbed.renderers.symbolic.hanabi import HanabiRenderer
from testbed.renderers.symbolic.harvest import HarvestRenderer
from testbed.renderers.symbolic.overcooked import OvercookedRenderer
from testbed.renderers.symbolic.werewolf import WerewolfRenderer
from testbed.renderers.textarena import TextArenaRenderer

_SYMBOLIC = {
    "beauty_contest": (BeautyContestAdapter, BeautyContestRenderer, BeautyContestParser),
    "gbs":       (GBSAdapter,       GBSRenderer,       GBSParser),
    "harvest":   (HarvestAdapter,   HarvestRenderer,   HarvestParser),
    "avalon":    (AvalonAdapter,    AvalonRenderer,    AvalonParser),
    "hanabi":    (HanabiAdapter,    HanabiRenderer,    HanabiParser),
    "werewolf":  (WerewolfAdapter,  WerewolfRenderer,  WerewolfParser),
    "overcooked": (OvercookedAdapter, OvercookedRenderer, OvercookedParser),
    **{variant: (partial(BoardAdapter, variant=variant), BoardRenderer, BoardParser)
       for variant in ("tictactoe", "connect_four", "gomoku")},
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

    if family == "meltingpot":  # game_id = substrate name; imports Melting Pot only when used
        from testbed.envs.meltingpot.adapter import MeltingPotAdapter
        from testbed.parsers.meltingpot import MeltingPotParser
        from testbed.renderers.meltingpot import MeltingPotRenderer
        env = MeltingPotAdapter(num_players=num_players, substrate=game_id, **env_kwargs)
        return env, MeltingPotRenderer(), MeltingPotParser()

    if family == "textarena":
        env = TextArenaAdapter(env_id=game_id, num_players=num_players, **env_kwargs)
        return env, TextArenaRenderer(), TextArenaParser()

    raise ValueError(f"Unknown game family: {family!r}")
