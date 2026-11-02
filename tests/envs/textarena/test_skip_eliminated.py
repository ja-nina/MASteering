"""skip_eliminated: after TextArena eliminates a player, the turn passes on instead of asking that player again."""
from types import SimpleNamespace

from testbed.envs.textarena.ta_adapter import TextArenaAdapter


class _StuckFFAEnv:
    """Mimics FFAMultiPlayerState: an invalid move never rotates; the 2nd one in a row eliminates the player."""
    def __init__(self, n=4):
        self.state = SimpleNamespace(num_players=n, current_player_id=0, elimination_order=[],
                                     error_count=0, made_invalid_move=False)
        self.asked = []

    def get_observation(self):
        self.asked.append(self.state.current_player_id)
        return self.state.current_player_id, "obs"

    def step(self, action):
        st = self.state
        if action == "bad":
            st.made_invalid_move = True
            if st.error_count < 1:
                st.error_count += 1
            else:
                st.elimination_order.append(st.current_player_id)
                st.error_count = 0
        else:
            st.current_player_id = (st.current_player_id + 1) % st.num_players
            while st.current_player_id in st.elimination_order:
                st.current_player_id = (st.current_player_id + 1) % st.num_players
        st.made_invalid_move = False
        return False, {}


def _play(skip, moves):
    env = _StuckFFAEnv()
    ad = TextArenaAdapter("PublicGoodsGame-v0", 4, _env=env, **({"skip_eliminated": True} if skip else {}))
    for m in moves:
        (aid, _), = ad.pending()
        ad.submit({aid: m})
    return env


def test_without_the_option_the_eliminated_player_is_asked_forever():
    env = _play(False, ["bad"] * 6)
    assert env.asked == [0] * 6 and env.state.elimination_order == [0, 0, 0]


def test_with_the_option_the_turn_passes_to_the_next_player_still_in():
    env = _play(True, ["bad", "bad", "ok", "ok", "ok", "ok"])
    assert env.asked == [0, 0, 1, 2, 3, 1]
    assert env.state.elimination_order == [0]


def test_option_is_not_passed_to_textarena():
    ad = TextArenaAdapter("PublicGoodsGame-v0", 4, skip_eliminated=True, seed=3)
    assert "skip_eliminated" not in ad._env_kwargs
