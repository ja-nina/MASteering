"""In-process HuggingFace policy with activation-steering forward hooks."""
from __future__ import annotations

from typing import Optional

from testbed.steering.activation import make_steering_hook
from testbed.types import SteeringSpec


def _resolve_submodule(model, dotted_name: str):
    """Resolve 'a.b.c' to a submodule of model."""
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


class _SteeringSession:
    """Context manager: register a forward hook on a submodule, remove on exit."""

    def __init__(self, model, layer_name: str, hook):
        self.module = _resolve_submodule(model, layer_name)
        self.hook = hook
        self.handle = None

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self.hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False


class TransformersPolicy:
    def __init__(self, model_id: str = "Qwen/Qwen3-4B",
                 temperature: float = 0.7, top_p: float = 0.8, top_k: int = 20,
                 max_new_tokens: int = 2048,
                 device: Optional[str] = None,
                 enable_thinking: bool = False,
                 steering: Optional[object] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # GTX 16xx / Turing cards don't support bfloat16; use float16 on CUDA
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        # the steering method (ActivationSteering) used to load vectors by agent
        self.steering = steering

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def _generate(self, inputs) -> str:
        import torch
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5),
                top_p=self.top_p, top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        self._last_truncated = (
            len(gen) >= self.max_new_tokens
            and gen[-1].item() != self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> str:
        inputs = self._build_inputs(system_prompt, user_prompt)
        if steering is not None and steering.method == "activation":
            if self.steering is None:
                raise ValueError("Activation steering requested but no steering "
                                 "method bound to the policy to load vectors.")
            vec = self.steering.load_vector(agent_id)
            hook = make_steering_hook(vec, coefficient=steering.coefficient)
            with _SteeringSession(self.model, steering.layer, hook):
                return self._generate(inputs)
        return self._generate(inputs)
