from PIL import Image

from testbed.registry import build_game
from testbed.visual import observe


def test_assassination_after_the_fifth_quest_renders():
    """Good wins on quest 5 (two failures first): the assassin's turn comes after the last quest."""
    env, renderer, _ = build_game(family="symbolic", game_id="avalon", num_players=6, env_kwargs={"seed": 0})
    env.reset()
    for vote in ("FAIL", "FAIL", "SUCCESS", "SUCCESS", "SUCCESS"):
        (leader, _), = env.pending()
        env.submit({leader: {"team": []}})                       # padded to the quest's size
        env.submit({a: "APPROVE" for a, _ in env.pending()})
        env.submit({a: vote for a, _ in env.pending()})
    assert env.context.extra["phase"] == "assassinate" and env.context.extra["quest_num"] == 5
    (assassin, obs), = env.pending()
    assert obs["quest_num"] == 5
    text, _ = observe(renderer, "text", obs, assassin, env.context)
    _, images = observe(renderer, "image", obs, assassin, env.context)
    assert "assassin" in text.lower() and isinstance(images[0], Image.Image)
    step = env.submit({assassin: "nobody"})
    assert step.done
