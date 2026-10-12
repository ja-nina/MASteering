"""Renderer for Melting Pot substrates: the agent's own view as an image (image mode only)."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext

SUBSTRATES = {
    "commons_harvest__open": (
        "You are one of several players collecting apples in an orchard. Walking onto an apple collects it "
        "(+1 reward). Apples regrow only near other apples: if a patch is picked empty it never grows back. "
        "FIRE_ZAP shoots a beam that freezes the player it hits for a while."),
    "commons_harvest__closed": (
        "You are one of several players collecting apples in orchard rooms with narrow entrances. Walking "
        "onto an apple collects it (+1). Apples regrow only near other apples. FIRE_ZAP freezes a hit player."),
    "clean_up": (
        "Apples (+1 when collected) grow in an orchard only while the nearby river is clean. Pollution "
        "builds up in the river over time; FIRE_CLEAN cleans the river tiles in front of you (no reward). "
        "FIRE_ZAP freezes a hit player. The group does well only if some players clean."),
    "prisoners_dilemma_in_the_matrix__repeated": (
        "Collect resources of two kinds (cooperate and defect). When you zap another player with "
        "FIRE_ZAP, you both get rewards from a prisoner's dilemma matrix depending on the resources you "
        "each hold. The game is repeated with the same partner."),
    "stag_hunt_in_the_matrix__repeated": (
        "Collect resources of two kinds (stag and hare). Zapping another player with FIRE_ZAP plays a stag "
        "hunt: both holding stag pays most, hare is the safe choice. The game is repeated."),
    "chicken_in_the_matrix__repeated": (
        "Collect resources of two kinds (hawk and dove). Zapping another player with FIRE_ZAP plays the "
        "game of chicken: hawk against hawk is worst for both. The game is repeated."),
    "territory__rooms": (
        "Claim wall tiles by painting them with your colour (FIRE_CLAIM); you earn reward for every tile "
        "you own over time. Other players can zap you (FIRE_ZAP) or destroy claimed walls."),
    "coins": (
        "Two players collect coins of their own and the other player's colour. Any coin gives you +1; "
        "a coin of your colour taken by the other player costs you 2."),
}

_ACTIONS = {
    "NOOP": "stay still", "FORWARD": "step forward (up in your view)", "BACKWARD": "step back",
    "STEP_LEFT": "step left", "STEP_RIGHT": "step right", "TURN_LEFT": "turn 90 degrees left",
    "TURN_RIGHT": "turn 90 degrees right", "FIRE_ZAP": "fire the zapping beam ahead of you",
    "FIRE_CLEAN": "fire the cleaning beam ahead of you", "FIRE_CLAIM": "fire the claiming beam ahead of you",
    "INTERACT": "interact with the object in front of you",
}


class MeltingPotRenderer:
    modes = ("image",)

    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        obs = raw_obs or {}
        actions = "\n".join(f"  {a} — {_ACTIONS.get(a, a.lower().replace('_', ' '))}"
                            for a in obs.get("action_names", []))
        return (
            f"You are {agent_id}, one of {obs.get('num_players', '?')} players in a 2D gridworld seen from above.\n\n"
            f"{SUBSTRATES.get(obs.get('substrate'), 'A Melting Pot multi-agent environment.')}\n\n"
            "Each turn you see your own view as an image. The view moves and turns with you: you are the "
            "avatar near the bottom centre, facing up, so FORWARD moves up in the image.\n"
            f"Plan your next {obs.get('plan_length', 4)} steps. Actions:\n{actions}\n\n"
            f"Respond in the form: ACTIONS: <action>, <action>, ... (1 to {obs.get('plan_length', 4)} actions)")

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        raise NotImplementedError("Melting Pot has image observations only")

    def render_visual(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs = raw_obs
        lines = [f"Step {obs['step']} of {obs['max_steps']}. Your reward so far: {obs['total_reward']:g} "
                 f"({obs['last_reward']:+g} during your last plan)."]
        if obs["last_plan"]:
            lines.append("Your last plan: " + ", ".join(obs["last_plan"]))
        lines += [f"{k.replace('_', ' ').lower()}: {v:g}" for k, v in obs["scalars"].items()]
        views = "your view" + (" (first image) and the whole map (second image)" if obs.get("world_rgb") is not None
                               else " (image)")
        lines.append(f"Look at {views} and plan your next steps. "
                     f"Respond in the form: ACTIONS: <action>, ... (1 to {obs['plan_length']} actions)")
        return "\n".join(lines)

    def render_images(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> list:
        from PIL import Image
        images = [Image.fromarray(raw_obs["rgb"])]
        if raw_obs.get("world_rgb") is not None:
            images.append(Image.fromarray(raw_obs["world_rgb"]))
        return images
