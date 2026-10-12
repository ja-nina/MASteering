# Observation modes (text vs. image)

The target models are vision-language models (Qwen3.5: `Qwen3_5ForConditionalGeneration`), so
a run can show the game state as a picture. A run's `observation` picks the mode:

| Mode | Prompt |
|---|---|
| `text` (default) | the usual text rendering |
| `image` | the state as image(s); the text keeps only what the picture does not show (dialogue, recent actions, the instruction) |
| `both` | the full text plus the image(s) |

The rules (system prompt) are always text. In a sweep, give a game `observation: [text, image]`
to run every condition in both modes (run ids get `__obs-image`).

| Game | Modes | What the image shows |
|---|---|---|
| gbs | text, image, both | group sum per round, coloured by feedback (picking variant: own guesses only) |
| beauty_contest | text, image, both | own guesses vs. each round's target |
| harvest | text, image, both | lake stock per month, catches, the stock now |
| werewolf | text, image, both | village ring: alive / eliminated with revealed roles, known roles |
| avalon | text, image, both | quest track, reject streak, players (leader, proposed team, known evil) |
| hanabi | text, image, both | the table: tokens, fireworks, others' cards, own cards with hints |
| overcooked | text, image, both | the kitchen grid with players, held items, pots |
| tictactoe / connect_four / gomoku | text, image, both | the board |
| Melting Pot | image | the agent's own pixel view (+ the map with `global_view`) |

## How images are run

- Images are scaled so the longer side is `image_size` (default 448 px; nearest-neighbour
  when upscaling pixel art). One Qwen3.5 image token covers 32×32 px, so 448×448 is ~200 tokens.
- The steer engine sends image turns as text with image placeholders plus `multi_modal_data`.
  Steering (`add_vector`) acts on every language-model position, image tokens included.
- The monitor rereads an image turn with the same images, then keeps the residual rows after
  the generation's prompt.
- `save_images: true` writes every turn's images next to the logs
  (`episode_N_images/t0003_player_1_0.png`); the JSONL records the paths.
- The engines accept `engine.max_images` images per prompt (default 2 when any run uses images).

Open questions this enables: the persona vectors were extracted from text-only prompts. Do they
still steer behaviour when the state comes as an image? Does the monitor read the same trait
expression in text and image games?
