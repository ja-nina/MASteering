"""Fast baseline policy: OpenAI-compatible client to a local vLLM server."""
from __future__ import annotations

from typing import Optional

from testbed.types import SteeringSpec


class VLLMPolicy:
    def __init__(self, model_id: str, endpoint: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", temperature: float = 0.7,
                 max_new_tokens: int = 256, _client=None) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        if _client is not None:
            self.client = _client
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=endpoint, api_key=api_key)

    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> tuple[str, bool]:
        if steering is not None and steering.method == "activation":
            raise ValueError("VLLMPolicy cannot apply activation steering; use "
                             "TransformersPolicy for activation runs.")
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=self.temperature, max_tokens=self.max_new_tokens)
        truncated = resp.choices[0].finish_reason == "length"
        return resp.choices[0].message.content, truncated
