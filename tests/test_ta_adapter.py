"""Tests for TextArenaAdapter, focusing on role discovery for role-aware steering."""
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
    roles = adapter.player_roles
    assert roles == {"player_0": "Mafia", "player_1": "Villager", "player_2": "Villager"}


def test_player_roles_skips_description_catalogue():
    """SecretMafia.roles is a role-description dict; player_roles should NOT return it."""
    class EnvOnlyHasDescriptionCatalogue:
        # Only 'roles' (descriptions), no 'player_roles'
        roles = {"Mafia": {"team": "Mafia"}, "Villager": {"team": "Village"}}
    adapter = _make_adapter(EnvOnlyHasDescriptionCatalogue())
    assert adapter.player_roles == {}


def test_player_roles_empty_when_env_none():
    adapter = TextArenaAdapter(env_id="X-v0", num_players=0, _env=None)
    assert adapter.player_roles == {}


def test_player_roles_empty_when_no_roles():
    adapter = _make_adapter(_FakeEnvNoRoles())
    assert adapter.player_roles == {}


def test_player_roles_falls_back_to_game_state():
    adapter = _make_adapter(_FakeEnvGameState())
    roles = adapter.player_roles
    assert roles == {"player_0": "Werewolf", "player_1": "Villager"}


def test_role_aware_steering_discovers_mafia_from_adapter():
    """End-to-end: RoleAwareActivationSteering correctly arms mafia agents."""
    from testbed.steering.role_aware import RoleAwareActivationSteering

    adapter = _make_adapter(_FakeMafiaEnv())
    steering = RoleAwareActivationSteering(
        target_roles=["Mafia"],
        default_config={
            "vector_path": "dummy.pt",
            "layer": "model.layers.20",
            "coefficient": 1.0,
        },
        mode="rotation",
    )
    steering.on_episode_start(adapter)

    assert "player_0" in steering._active_agents, "Mafia agent should be armed"
    assert "player_1" not in steering._active_agents
    assert "player_2" not in steering._active_agents

    spec = steering.steering_spec("player_0")
    assert spec is not None
    assert spec.mode == "rotation"
    assert spec.coefficient == 1.0

    # Non-mafia players get noop
    assert steering.steering_spec("player_1") is None
    assert steering.steering_spec("player_2") is None


def test_role_aware_steering_no_roles_emits_warning():
    """When roles can't be discovered, a warning is emitted and no agents are armed."""
    import warnings
    from testbed.steering.role_aware import RoleAwareActivationSteering

    adapter = _make_adapter(_FakeEnvNoRoles())
    steering = RoleAwareActivationSteering(
        target_roles=["Mafia"],
        default_config={"vector_path": "dummy.pt", "layer": "model.layers.20"},
        mode="rotation",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        steering.on_episode_start(adapter)

    assert len(steering._active_agents) == 0
    assert any("could not discover roles" in str(warning.message) for warning in w)
