"""Shared generation utilities for rollout and vllm_rollout."""
from __future__ import annotations
import re

STRUCTURED_FORMAT_INSTRUCTION = (
    "\n\nAlways respond in this exact format:\n"
    "<strategy>\n"
    "Think carefully through your position, what the other player wants, and what move "
    "serves your goals best. The more thoroughly you reason here, the better your decisions.\n"
    "</strategy>\n"
    "<action>\n"
    "What you actually want to say to the other player (an optimal action is between 100-200 words).\n"
    "</action>"
)


def _has_action_tag(text: str) -> bool:
    """Return True if the model opened an <action> block (closing tag optional)."""
    return bool(re.search(r"<action>", text))


def _extract_action(text: str) -> str:
    """Extract the action content from structured output.

    Accepts both closed (<action>…</action>) and unclosed (<action>…EOF) forms —
    the model often omits the closing tag without it being a meaningful error.
    Falls back to stripping <think> blocks when no <action> tag is present at all.
    """
    # Prefer the closed form
    m = re.search(r"<action>(.*?)</action>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Accept unclosed: take everything after <action>
    m2 = re.search(r"<action>(.*)", text, flags=re.DOTALL)
    if m2:
        return m2.group(1).strip()
    # Last resort: strip native <think> blocks
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
