"""Config parsing + builders for steering, probing, and policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from testbed.steering.activation import ActivationSteering
from testbed.steering.noop import NoOpSteering


def build_steering(cfg: Dict[str, Any]):
    method = cfg.get("default", "noop")
    per_agent = cfg.get("per_agent", {}) or {}
    default_config = cfg.get("default_config") or None
    mode = cfg.get("mode", "adaptive")

    if method == "noop":
        return NoOpSteering()
    if method == "activation":
        return ActivationSteering(per_agent=per_agent,
                                  default_config=default_config,
                                  mode=mode)
    if method == "role_aware_activation":
        from testbed.steering.role_aware import RoleAwareActivationSteering
        target_roles = cfg.get("target_roles") or []
        if default_config is None:
            raise ValueError(
                "role_aware_activation requires 'default_config' in the steering block"
            )
        return RoleAwareActivationSteering(
            target_roles=target_roles,
            default_config=default_config,
            mode=mode,
        )
    if method == "svd":
        from testbed.steering.svd_steering import SVDSteering
        return SVDSteering(
            basis_path=cfg["basis_path"],
            per_agent=cfg.get("per_agent", {}),
            default_config=cfg.get("default_config"),
            mode=cfg.get("mode", "additive"),
        )
    raise ValueError(f"Unknown steering method: {method!r}")


def build_probe(probing_cfg: Optional[Dict[str, Any]]):
    """Build a PersonaProbe if probing is enabled in the config, else None."""
    if not probing_cfg or not probing_cfg.get("enabled"):
        return None

    if probing_cfg.get("mode") == "svd":
        from testbed.probing.svd_probe import SVDPersonaProbe
        return SVDPersonaProbe(
            basis_path=probing_cfg["basis_path"],
            layers=probing_cfg.get("layers", [18]),
            hook=probing_cfg.get("hook", "attn"),
            layer_path_template=probing_cfg.get("layer_path_template", "model.layers.{}"),
            window_tokens=probing_cfg.get("window_tokens", 10),
            top_k=probing_cfg.get("top_k", 5),
        )

    from testbed.probing.persona_probe import PersonaProbe
    template = probing_cfg.get("layer_path_template", "model.layers.{}")

    # Prefer `layers: [10, 20, 29]` (multi-layer); fall back to `layer: N` (legacy).
    if "layers" in probing_cfg:
        return PersonaProbe(
            vectors_dir=probing_cfg["vectors_dir"],
            layers=probing_cfg["layers"],
            layer_path_template=template,
            window_tokens=probing_cfg.get("window_tokens", 10),
            top_k=probing_cfg.get("top_k", 5),
        )
    layer = probing_cfg.get("layer", 20)
    return PersonaProbe(
        vectors_dir=probing_cfg["vectors_dir"],
        layer=layer,
        layer_path=template.format(layer),
        window_tokens=probing_cfg.get("window_tokens", 10),
        top_k=probing_cfg.get("top_k", 5),
    )


_TRANSFORMERS_STRUCTURAL = {
    "backend", "model_id", "device", "enable_thinking", "reasoning_cue",
}


def build_policy(model_cfg: Dict[str, Any], steering=None,
                 probing_cfg: Optional[Dict[str, Any]] = None):
    backend = model_cfg.get("backend", "transformers")
    model_id = model_cfg["model_id"]
    if backend == "transformers":
        from testbed.policy.transformers_policy import TransformersPolicy
        gen_kwargs = {k: v for k, v in model_cfg.items()
                      if k not in _TRANSFORMERS_STRUCTURAL}
        return TransformersPolicy(
            model_id=model_id,
            device=model_cfg.get("device"),
            enable_thinking=model_cfg.get("enable_thinking", False),
            reasoning_cue=model_cfg.get("reasoning_cue", False),
            steering=steering,
            probe=build_probe(probing_cfg),
            **gen_kwargs,
        )
    if backend == "vllm":
        import os
        from testbed.policy.vllm_policy import VLLMPolicy
        api_key = model_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "EMPTY")
        return VLLMPolicy(
            model_id=model_id,
            endpoint=model_cfg.get("endpoint", "http://localhost:8000") + "/v1",
            temperature=model_cfg.get("temperature", 0.7),
            api_key=api_key)
    raise ValueError(f"Unknown backend: {backend!r}")


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
    probing: Dict[str, Any]
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
            probing=d.get("probing", {"enabled": False}),
            logging_dir=d.get("logging", {}).get("dir", "logs/"),
            env_kwargs=game.get("env_kwargs", {}),
        )
