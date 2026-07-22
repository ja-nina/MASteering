"""Map a game spec to its (adapter, renderer, parser) triple."""
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

_SYMBOLIC = {
    "beauty_contest": (BeautyContestAdapter, BeautyContestRenderer, BeautyContestParser),
    "gbs": (GBSAdapter, GBSRenderer, GBSParser),
    # Faithful replication of Riedl (2025, arXiv 2510.05174) — mechanically
    # identical to gbs, distinguished only by env_kwargs (hide_group_size=True,
    # feedback="directional", persona_mode, personas list).
    "gbs_exact_replication": (GBSAdapter, GBSRenderer, GBSParser),
}


def build_game(*, family: str, game_id: str, num_players: int,
               env_kwargs: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    if family == "symbolic":
        if game_id not in _SYMBOLIC:
            raise ValueError(f"Unknown symbolic game: {game_id}")
        AdapterCls, RendererCls, ParserCls = _SYMBOLIC[game_id]
        env = AdapterCls(num_players=num_players, **env_kwargs)
        return env, RendererCls(), ParserCls()
    if family == "textarena":
        env = TextArenaAdapter(env_id=game_id, num_players=num_players, **env_kwargs)
        return env, TextArenaRenderer(), TextArenaParser()
    raise ValueError(f"Unknown game family: {family}")
