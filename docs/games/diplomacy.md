# Diplomacy

**Family:** `textarena` | **ID:** `Diplomacy-v0` | **Status:** fully playable via TextArena

---

## Game overview

Diplomacy is a 7-player strategic negotiation game set in pre-WWI Europe.
Players control one of seven Great Powers, write orders for their armies and
fleets, and negotiate alliances — then all orders resolve simultaneously.
There is no luck: the game is decided entirely by negotiation, deception,
and strategic planning.

Key properties for LLM research:
- **Extended multi-turn negotiation**: agents send private messages to each other
  before each order phase
- **Simultaneous action revelation**: all orders become public only after commitment,
  rewarding deception
- **Coalition dynamics**: players must form alliances they can credibly betray

---

## Status: fully working via TextArena

Diplomacy is provided by the `textarena` library as `Diplomacy-v0`.
No custom adapter, renderer, or parser is needed — the existing
`TextArenaAdapter`/`TextArenaRenderer`/`TextArenaParser` triad handles it.

```yaml
game:
  family: textarena
  id: Diplomacy-v0
```

---

## Quick start

```bash
python scripts/run_episode.py \
  --config config/games/diplomacy/diplomacy_noop_5p.yaml \
  --episodes 1
```

Logs land in `logs/diplomacy/<run_id>/`.

---

## Config keys

TextArena handles all game parameters internally.  Pass additional kwargs via
`env_kwargs` if the TextArena `Diplomacy-v0` environment accepts them (check
the TextArena documentation for supported keys).

---

## Custom adapter (optional)

If you need finer-grained control — structured order parsing, private message
logging, per-player alliance tracking — you can write a custom symbolic adapter
instead of using the TextArena wrapper.  Follow the pattern in
`testbed/envs/symbolic/avalon.py` and register as `"diplomacy_custom"` in
`testbed/registry.py`.

---

## Key research questions

- Can LLM agents sustain multi-message deception while maintaining internal
  consistency (not contradicting past messages)?
- Do ToM-steered agents form more successful alliances, or does explicit other-
  modelling make deception harder?
- Does a `machiavellian` persona (from `config/shared/behavioral_personas.yaml`)
  outperform a `cooperative` persona in final supply-centre count?
- How does reasoning effort (thinking tokens) affect order quality?
