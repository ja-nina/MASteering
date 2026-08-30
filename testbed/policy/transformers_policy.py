"""In-process HuggingFace policy with activation-steering and persona-probe hooks."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

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
                 model=None,
                 tokenizer=None,
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
        model             — optional pre-built model object; when provided the
                            model is used as-is (caller owns eval/train mode).
        tokenizer         — optional pre-built tokenizer; loaded from model_id
                            if omitted.
        All remaining kwargs are forwarded to model.generate().
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.reasoning_cue = reasoning_cue and not enable_thinking

        disable_quantization = gen_kwargs.pop("disable_quantization", False)

        if model is not None:
            # Injected model — caller is responsible for eval/train mode.
            self.model = model
            self.device = device or str(next(model.parameters()).device)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
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

    def _generate_with_full_ids(self, inputs) -> tuple[str, "torch.Tensor"]:
        """Two-phase constrained generation for GRPO rollout.

        Phase 1: append a forced <strategy> token, generate until </strategy>
                 is detected, then stop.
        Phase 2: inject \\n<action> into the context and generate the action body.

        Returns (text, (full_ids_cpu, input_len)) where input_len is the length
        of the original prompt + the forced <strategy> prefix token(s), so that
        log-prob computation in grpo.py covers strategy body + injected <action>
        tag + action body — i.e. everything the model committed to.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        temperature = self._gen_kwargs.get("temperature", 0.7)
        kwargs = {
            **self._gen_kwargs,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        input_ids = inputs["input_ids"]   # [1, prompt_len]
        attn_mask  = inputs.get("attention_mask")

        # ── Phase 1: force <strategy>, generate until </strategy> ─────────────
        strat_open_ids = torch.tensor(
            [self.tokenizer.encode("<strategy>", add_special_tokens=False)],
            dtype=torch.long, device=input_ids.device,
        )
        p1_ids = torch.cat([input_ids, strat_open_ids], dim=1)
        if attn_mask is not None:
            attn_ext = torch.ones(1, strat_open_ids.shape[1],
                                  dtype=attn_mask.dtype, device=attn_mask.device)
            p1_mask = torch.cat([attn_mask, attn_ext], dim=1)
        else:
            p1_mask = None
        p1_len = p1_ids.shape[1]

        class _StopOnClose(StoppingCriteria):
            def __call__(self_inner, ids, scores, **kw):
                tail = self.tokenizer.decode(ids[0, p1_len:], skip_special_tokens=True)
                return "</strategy>" in tail

        p1_inputs = {"input_ids": p1_ids}
        if p1_mask is not None:
            p1_inputs["attention_mask"] = p1_mask

        with torch.no_grad():
            out1 = self.model.generate(
                **p1_inputs, **kwargs,
                stopping_criteria=StoppingCriteriaList([_StopOnClose()]),
            )

        strat_gen_ids = out1[0, p1_len:]   # [strat_gen_len]
        strat_text = self.tokenizer.decode(strat_gen_ids, skip_special_tokens=True)
        if "</strategy>" not in strat_text:
            close_ids = torch.tensor(
                self.tokenizer.encode("\n</strategy>", add_special_tokens=False),
                dtype=torch.long, device=input_ids.device,
            )
            strat_gen_ids = torch.cat([strat_gen_ids, close_ids])
            strat_text = self.tokenizer.decode(strat_gen_ids, skip_special_tokens=True)

        # ── Phase 2: inject \n<action>, generate action body ──────────────────
        action_tag_ids = torch.tensor(
            [self.tokenizer.encode("\n<action>", add_special_tokens=False)],
            dtype=torch.long, device=input_ids.device,
        )
        p2_ids = torch.cat([p1_ids, strat_gen_ids.unsqueeze(0), action_tag_ids], dim=1)
        p2_len = p2_ids.shape[1]
        if p1_mask is not None:
            p2_mask = torch.ones(1, p2_ids.shape[1],
                                 dtype=p1_mask.dtype, device=p1_mask.device)
            p2_inputs = {"input_ids": p2_ids, "attention_mask": p2_mask}
        else:
            p2_inputs = {"input_ids": p2_ids}

        with torch.no_grad():
            out2 = self.model.generate(**p2_inputs, **kwargs)

        action_gen_ids = out2[0, p2_len:]
        action_text = self.tokenizer.decode(action_gen_ids, skip_special_tokens=True)

        full_text = f"<strategy>{strat_text}\n<action>{action_text}"
        full_ids  = out2[0:1].cpu()   # [1, p2_len + action_len]
        # input_len = p1_len: log probs cover strategy body + \n<action> + action body
        return full_text, (full_ids, p1_len)

    def _generate_with_logprob(self, inputs) -> tuple[str, "torch.Tensor"]:
        """Generate a response and compute its log-probability via teacher forcing.

        The sampling step runs under torch.no_grad() (selecting the action).
        A second forward pass with the full sequence then computes log P(action)
        with gradient tracking so REINFORCE can backprop through LoRA params.
        """
        import torch
        import torch.nn.functional as F

        temperature = self._gen_kwargs.get("temperature", 0.7)
        max_new_tokens = self._gen_kwargs.get("max_new_tokens", 1024)
        kwargs = {
            **self._gen_kwargs,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # 1. Sample the action (no gradient needed here).
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)

        gen = out[0, input_len:]   # generated token ids, shape [gen_len]
        gen_len = gen.shape[0]

        # 2. Decode text.
        if self.enable_thinking or self.reasoning_cue:
            text = self.tokenizer.decode(gen, skip_special_tokens=False)
            eos = self.tokenizer.eos_token or ""
            text = text.rstrip().removesuffix(eos).rstrip()
        else:
            text = self.tokenizer.decode(gen, skip_special_tokens=True)

        # 3. Teacher-forcing forward pass WITH gradient tracking.
        if gen_len == 0:
            log_prob = torch.tensor(0.0)
        else:
            full_ids = out[0:1]  # [1, input_len + gen_len]
            # No no_grad here — we want gradients through LoRA params.
            logits = self.model(full_ids).logits  # [1, T, vocab]
            # Logit at position (input_len-1 + i) predicts generated token i.
            gen_logits = logits[0, input_len - 1 : input_len - 1 + gen_len, :]
            log_probs_matrix = F.log_softmax(gen_logits, dim=-1)
            token_log_probs = log_probs_matrix[torch.arange(gen_len, device=gen.device), gen]
            log_prob = token_log_probs.sum()

        return text, log_prob

    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec],
            return_logprob: bool = False,
            return_full_ids: bool = False) -> tuple[str, Any]:
        """Generate an action.

        Args:
            return_logprob:  (legacy) returns (text, log_prob_tensor) with grad.
            return_full_ids: returns (text, full_ids_cpu_tensor) with NO grad.
                             Use this during rollout collection; call
                             recompute_logprobs() separately before backward.

        Returns:
            (action_str, full_ids_cpu)   if return_full_ids=True
            (action_str, log_prob)       if return_logprob=True
            (action_str, truncated_bool) otherwise
        """
        inputs = self._build_inputs(system_prompt, user_prompt)

        hooks = []
        if self.steering is not None and hasattr(self.steering, "apply_hooks"):
            hooks.extend(self.steering.apply_hooks(agent_id, self.model))
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

        if return_full_ids:
            _gen = self._generate_with_full_ids
        elif return_logprob:
            _gen = self._generate_with_logprob
        else:
            _gen = self._generate

        if hooks:
            with _HookSession(self.model, hooks):
                result = _gen(inputs)
        else:
            result = _gen(inputs)

        self._last_probe = get_scores() if get_scores is not None else {}
        return result

    def probe_text(self, text: str) -> "Dict[str, Any]":
        """Single forward pass on text with probe hooks active; no generation.

        Runs with all LoRA adapters disabled so this always reflects base-model
        weights (i.e. the frozen opponent's perspective).  Used to measure the
        opponent's hidden-state reaction to each of the K candidate trainee
        responses without running a full generate() call.

        Returns the same dict format as self._last_probe.
        """
        import torch
        if self.probe is None:
            return {}
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        probe_hooks, get_scores = self.probe.make_hook()
        with _HookSession(self.model, probe_hooks):
            with self.model.disable_adapter():
                with torch.no_grad():
                    self.model(**inputs)
        return get_scores() if get_scores is not None else {}

    def probe_response_in_context(
        self, system_prompt: str, user_prompt: str, response: str
    ) -> "Dict[str, Any]":
        """Probe base-model activations averaged over response tokens only.

        Builds [system, user, assistant(response)], computes the context boundary
        (system + user tokens), then averages hidden states over only the response
        token positions — not the system prompt or user observation.
        """
        import torch
        if self.probe is None:
            return {}

        # Context boundary: tokenize without the response to find where response starts
        context_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        context_text = self.tokenizer.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=True
        )
        context_len = self.tokenizer(
            context_text, return_tensors="pt"
        )["input_ids"].shape[1]

        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": response},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        # Custom hooks that average only over response token positions
        layer_results: "Dict[str, dict]" = {}
        hooks_handles = []

        def _make_hook(layer_key: str):
            def _h(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                resp_h = h[0, context_len:, :].detach().float()  # [resp_len, d]
                if resp_h.shape[0] == 0:
                    return
                mean_h = resp_h.mean(dim=0)  # [d]
                probe = self.probe
                layer_int = int(layer_key)
                if probe._use_direct:
                    M     = probe._M.get(layer_int)
                    Mnorm = probe._M_norms.get(layer_int)
                    if M is None:
                        return
                    h_norm = mean_h.norm().clamp(min=1e-8)
                    z = (M.to(mean_h.device) @ mean_h) / (Mnorm.to(mean_h.device) * h_norm)
                else:
                    Vk = probe._Vk.get(layer_int)
                    if Vk is None:
                        return
                    z = Vk.to(mean_h.device) @ mean_h
                layer_results[layer_key] = {"z": z.tolist()}
            return _h

        for layer_int in self.probe.layers:
            path = self.probe._layer_path(layer_int)
            # resolve submodule
            mod = self.model
            for part in path.split("."):
                mod = getattr(mod, part)
            handle = mod.register_forward_hook(_make_hook(str(layer_int)))
            hooks_handles.append(handle)

        try:
            with self.model.disable_adapter():
                with torch.no_grad():
                    self.model(**inputs)
        finally:
            for h in hooks_handles:
                h.remove()

        return layer_results

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        if self.reasoning_cue:
            text += "<think>\n"
        return self.tokenizer(text, return_tensors="pt").to(self.device)
