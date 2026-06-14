"""Config parsing + builders for steering and policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from testbed.steering.activation import ActivationSteering
from testbed.steering.noop import NoOpSteering
from testbed.steering.prompt_injection import PromptInjectionSteering


def build_steering(cfg: Dict[str, Any]):
    method = cfg.get("default", "noop")
    per_agent = cfg.get("per_agent", {}) or {}
    if method == "noop":
        return NoOpSteering()
    if method == "prompt_injection":
        return PromptInjectionSteering(per_agent=per_agent)
    if method == "activation":
        return ActivationSteering(per_agent=per_agent)
    raise ValueError(f"Unknown steering method: {method}")


def build_policy(model_cfg: Dict[str, Any], steering=None):
    backend = model_cfg.get("backend", "transformers")
    model_id = model_cfg["model_id"]
    if backend == "transformers":
        from testbed.policy.transformers_policy import TransformersPolicy
        return TransformersPolicy(
            model_id=model_id,
            temperature=model_cfg.get("temperature", 0.7),
            steering=steering)
    if backend == "vllm":
        from testbed.policy.vllm_policy import VLLMPolicy
        return VLLMPolicy(
            model_id=model_id,
            endpoint=model_cfg.get("endpoint", "http://localhost:8000") + "/v1",
            temperature=model_cfg.get("temperature", 0.7))
    raise ValueError(f"Unknown backend: {backend}")


@dataclass
class RunConfig:
    run_id: str
    game_family: str
    game_id: str
    episodes: int
    num_players: Optional[int]
    max_parse_retries: int
    model: Dict[str, Any]
    steering: Dict[str, Any]
    logging_dir: str
    env_kwargs: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        game = d["game"]
        agents = d.get("agents", {})
        return cls(
            run_id=d["run_id"],
            game_family=game["family"],
            game_id=game["id"],
            episodes=d.get("episodes", 1),
            num_players=agents.get("count"),
            max_parse_retries=agents.get("max_parse_retries", 5),
            model=d.get("model", {}),
            steering=d.get("steering", {"default": "noop", "per_agent": {}}),
            logging_dir=d.get("logging", {}).get("dir", "logs/"),
            env_kwargs=game.get("env_kwargs", {}),
        )
