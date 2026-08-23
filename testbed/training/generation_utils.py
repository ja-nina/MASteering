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
    "What you actually say to the other player (20-100 words).\n"
    "</action>"
)


def _has_action_tag(text: str) -> bool:
    """Return True only if the model produced a well-formed <action>…</action> block."""
    return bool(re.search(r"<action>.*?</action>", text, flags=re.DOTALL))


def _extract_action(text: str) -> str:
    """Extract the <action>...</action> block from structured output.

    Falls back to the full text (stripped of any <think> blocks) if the
    model did not follow the format.
    """
    m = re.search(r"<action>(.*?)</action>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
