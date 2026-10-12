"""Tests for TextArenaAdapter role discovery (used by role-targeted steering)."""
from testbed.envs.textarena.ta_adapter import TextArenaAdapter


class _FakeMafiaEnv:
    """Minimal stub of SecretMafiaEnv after reset()."""
    # Role-description catalogue (static, same across all games)
    roles = {
        "Villager": {"team": "Village", "description": "..."},
        "Mafia": {"team": "Mafia", "description": "..."},
    }
    # Per-seat role assignment (set during reset)
    player_roles = {0: "Mafia", 1: "Villager", 2: "Villager"}


class _FakeEnvNoRoles:
    """Env that exposes neither player_roles nor roles."""
    pass


class _FakeEnvGameState:
    """Env that stores roles inside state.game_state (alternative pattern)."""
    class _State:
        game_state = {"player_roles": {0: "Werewolf", 1: "Villager"}}
    state = _State()


def _make_adapter(_env):
    return TextArenaAdapter(env_id="TestGame-v0", num_players=0, _env=_env)


def test_player_roles_returns_normalised_dict():
    adapter = _make_adapter(_FakeMafiaEnv())
    roles = adapter.roles()
    assert roles == {"player_0": "Mafia", "player_1": "Villager", "player_2": "Villager"}


def test_player_roles_skips_description_catalogue():
    """SecretMafia.roles is a role-description dict; player_roles should NOT return it."""
    class EnvOnlyHasDescriptionCatalogue:
        # Only 'roles' (descriptions), no 'player_roles'
        roles = {"Mafia": {"team": "Mafia"}, "Villager": {"team": "Village"}}
    adapter = _make_adapter(EnvOnlyHasDescriptionCatalogue())
    assert adapter.roles() == {}


def test_player_roles_empty_when_env_none():
    adapter = TextArenaAdapter(env_id="X-v0", num_players=0, _env=None)
    assert adapter.roles() == {}


def test_player_roles_empty_when_no_roles():
    adapter = _make_adapter(_FakeEnvNoRoles())
    assert adapter.roles() == {}


def test_player_roles_falls_back_to_game_state():
    adapter = _make_adapter(_FakeEnvGameState())
    roles = adapter.roles()
    assert roles == {"player_0": "Werewolf", "player_1": "Villager"}


def test_teams_marks_deceiving_side_evil():
    adapter = _make_adapter(_FakeMafiaEnv())
    assert adapter.teams() == {"player_0": "evil", "player_1": "good", "player_2": "good"}
