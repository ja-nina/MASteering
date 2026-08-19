"""In-process HuggingFace policy with activation-steering and persona-probe hooks."""
from __future__ import annotations

from typing import Any, Dict, Optional

from testbed.steering.activation import make_steering_hook
from testbed.types import SteeringSpec

_GEN_DEFAULTS: Dict[str, Any] = {
    "max_new_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
}


def _resolve_submodule(model, dotted_name: str):
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


class _HookSession:
    """Context manager: register one or more forward hooks, remove all on exit."""

    def __init__(self, model, hooks: list):
        """
        hooks — list of (layer_dotpath, hook_fn) pairs
        """
        self._model = model
        self._specs = hooks
        self._handles = []

    def __enter__(self):
        for layer_path, hook_fn in self._specs:
            module = _resolve_submodule(self._model, layer_path)
            self._handles.append(module.register_forward_hook(hook_fn))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        return False


class TransformersPolicy:
    def __init__(self, model_id: str = "Qwen/Qwen3-4B",
                 device: Optional[str] = None,
                 enable_thinking: bool = False,
                 reasoning_cue: bool = False,
                 steering: Optional[object] = None,
                 probe: Optional[object] = None,
                 **gen_kwargs) -> None:
        """
        enable_thinking   — use Qwen3 native thinking mode (<think> tokens).
        reasoning_cue     — prime the assistant turn with '<think>\\n' so the
                            model reasons in a closed scope before answering,
                            without enabling native thinking mode. Ignored when
                            enable_thinking is True.
        steering          — ActivationSteering instance (for load_vector()).
        probe             — PersonaProbe instance; if set, projects every
                            completion's hidden states onto all trait vectors
                            and stores results in self._last_probe.
        All remaining kwargs are forwarded to model.generate().
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.reasoning_cue = reasoning_cue and not enable_thinking
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        disable_quantization = gen_kwargs.pop("disable_quantization", False)
        load_kwargs = {"quantization_config": None} if disable_quantization else {}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, **load_kwargs).to(self.device)
        self.model.eval()
        self.steering = steering
        self.probe = probe
        self._gen_kwargs: Dict[str, Any] = {**_GEN_DEFAULTS, **gen_kwargs}
        self._last_probe: Dict[str, float] = {}

    def _generate(self, inputs) -> tuple[str, bool]:
        import torch

        temperature = self._gen_kwargs.get("temperature", 0.7)
        max_new_tokens = self._gen_kwargs.get("max_new_tokens", 1024)
        kwargs = {
            **self._gen_kwargs,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)

        gen = out[0][inputs["input_ids"].shape[1]:]
        truncated = (
            len(gen) >= max_new_tokens
            and gen[-1].item() != self.tokenizer.eos_token_id
        )

        if self.enable_thinking or self.reasoning_cue:
            text = self.tokenizer.decode(gen, skip_special_tokens=False)
            eos = self.tokenizer.eos_token or ""
            return text.rstrip().removesuffix(eos).rstrip(), truncated
        return self.tokenizer.decode(gen, skip_special_tokens=True), truncated

    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> tuple[str, bool]:
        inputs = self._build_inputs(system_prompt, user_prompt)

        # Build the list of hooks to register: steering (if active) + probe (always)
        hooks = []

        # SVDSteering path — duck-typed: any steering object with apply_hooks()
        if self.steering is not None and hasattr(self.steering, "apply_hooks"):
            svd_hooks = self.steering.apply_hooks(agent_id, self.model)
            hooks.extend(svd_hooks)
        elif steering is not None and steering.method == "activation":
            if self.steering is None:
                raise ValueError("Activation steering requested but no steering "
                                 "method bound to the policy.")
            vec = self.steering.load_vector(agent_id)
            steer_hook = make_steering_hook(
                vec, coefficient=steering.coefficient, mode=steering.mode
            )
            hooks.append((steering.layer, steer_hook))

        get_scores = None
        if self.probe is not None:
            probe_hooks, get_scores = self.probe.make_hook()
            hooks.extend(probe_hooks)

        if hooks:
            with _HookSession(self.model, hooks):
                result = self._generate(inputs)
        else:
            result = self._generate(inputs)

        self._last_probe = get_scores() if get_scores is not None else {}
        return result

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        if self.reasoning_cue:
            text += "<think>\n"
        return self.tokenizer(text, return_tensors="pt").to(self.device)
