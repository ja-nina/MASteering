"""vLLM-Hook (MIA) engines; the only module that imports vllm / vLLM-Hook (cluster only).

Model specs, sampling settings, chat template and MIA helpers come from PersVecFast
(installed in the same env), so games run the model exactly as its vectors were made.

steer engine:   worker `steer`, built-in add_vector op (stays inside vLLM's CUDA graphs).
                Every request carries its own steering config; unsteered requests carry a
                zero vector with coefficient 0 (as in PersVecFast). Prefix caching on: game
                prompts share long prefixes across turns.
capture engine: worker `capture_hs` at the monitor layers only. Reads a finished turn by
                prefilling prompt + completion tokens (max_tokens=1, hooks_on=prefill,
                hs_mode=all_tokens) and keeps the completion's rows. vLLM-Hook cannot
                capture in the steer engine, so this is a readout by the unsteered model.

Image observations (Qwen3.5 is a vision-language model): the user turn gets one image
placeholder per image and the images go in as multi_modal_data; steering and capture act on
the language-model blocks, image tokens included. The readout re-sends the images with
prompt + completion text and keeps the rows after the generation's (expanded) prompt length.

Both engines live at once: on separate GPUs (`gpus`) or sharing one (`memory` split).
"""
from __future__ import annotations

import gc
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch

from persvecfast.mia import clear_delivered, layer_key  # also sets PersVecFast's MIA/vLLM env defaults
from persvecfast.models import MODELS, ModelSpec, model_shape, sampling_kwargs
from persvecfast.prompts import chat_prompt

from testbed.engine import TurnRequest, TurnResult
from testbed.util import log, save_pt, write_json

CAPTURE_BATCH = 32  # turns read per capture call (bounds the captured rows held in RAM)


class CaptureMismatch(RuntimeError):
    pass


def template(tokenizer, spec: ModelSpec, system: str, user: str, n_images: int) -> str:
    if not n_images:
        return chat_prompt(tokenizer, system, user, spec.enable_thinking)
    content = [{"type": "image"}] * n_images + [{"type": "text", "text": user}]
    kwargs = {} if spec.enable_thinking is None else {"enable_thinking": spec.enable_thinking}
    return tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True, **kwargs)


class SteerEngine:
    def __init__(self, llm, spec: ModelSpec, tokenizer, zero_vector: str, sampling: dict,
                 max_model_len: int, pixels_per_token: int = 32) -> None:
        self.llm, self.spec, self.tokenizer = llm, spec, tokenizer
        self.zero_vector, self.sampling, self.max_model_len = zero_vector, sampling, max_model_len
        self.pixels_per_token = pixels_per_token  # vision patch size x spatial merge

    def _image_tokens(self, images) -> int:
        p = self.pixels_per_token
        return sum(-(-im.width // p) * -(-im.height // p) + 2 for im in images)  # + start/end markers

    def _steer_arg(self, steer) -> dict:
        if steer is None or steer.vector_path is None:  # unsteered, or a prompt persona only
            return {"method": "add_vector", "coefficient": 0.0, "optimal_layer": 0, "vector_path": self.zero_vector}
        return {"method": "add_vector", "coefficient": steer.coeff, "optimal_layer": steer.layer,
                "vector_path": steer.vector_path}

    def generate(self, requests: List[TurnRequest]) -> List[TurnResult]:
        """One batched call. Generation is capped so prompt + completion (+1 for the monitor's
        readout pass) fits max_model_len; a prompt that leaves no room comes back empty with
        finish_reason "prompt_too_long" (the parser then falls back) instead of failing the batch."""
        from vllm import SamplingParams

        results: Dict[tuple, TurnResult] = {}
        prompts, params, sent = [], [], []
        texts = []
        for r in requests:
            text = template(self.tokenizer, self.spec, r.system, r.user, len(r.images))
            ids = self.tokenizer.encode(text, add_special_tokens=False)  # the template has them
            room = self.max_model_len - len(ids) - self._image_tokens(r.images) - 1
            if room < 16:
                results[r.key] = TurnResult(r.key, "", "prompt_too_long", ids, [], text, r.images)
                continue
            kwargs = {**sampling_kwargs(self.spec, r.seed, r.max_tokens), **self.sampling}
            kwargs["max_tokens"] = min(kwargs["max_tokens"], room)
            # image prompts go in as text: vLLM expands each placeholder to the image's tokens
            prompts.append({"prompt": text, "multi_modal_data": {"image": list(r.images)}} if r.images
                           else {"prompt_token_ids": ids})
            params.append(SamplingParams(**kwargs, extra_args={"steer": self._steer_arg(r.steer)}))
            sent.append(r)
            texts.append(text)
        for r, text, o in zip(sent, texts, self.llm.generate(prompts, params) if prompts else []):
            c = o.outputs[0]
            results[r.key] = TurnResult(r.key, c.text, c.finish_reason, list(o.prompt_token_ids),
                                        list(c.token_ids), text, r.images)
        return [results[r.key] for r in requests]


class CaptureEngine:
    def __init__(self, llm, spec: ModelSpec, layers: List[int], hidden: int) -> None:
        self.llm, self.spec, self.layers, self.hidden = llm, spec, list(layers), hidden

    def read(self, results: List[TurnResult]) -> List[Dict[int, torch.Tensor]]:
        rows: List[Optional[Dict[int, torch.Tensor]]] = [None] * len(results)
        todo = [i for i, r in enumerate(results) if r.token_ids]
        for start in range(0, len(todo), CAPTURE_BATCH):
            batch = todo[start:start + CAPTURE_BATCH]
            for i, r in zip(batch, self._read([results[i] for i in batch])):
                rows[i] = r
        empty = {l: torch.zeros(0, self.hidden) for l in self.layers}
        return [r if r is not None else empty for r in rows]

    def _read(self, results: List[TurnResult]) -> List[Dict[int, torch.Tensor]]:
        from vllm import SamplingParams

        capture = {"hooks_on": "prefill", "hs_mode": "all_tokens"}
        # text turns: the exact generated tokens; image turns: prompt + completion text with the images
        # again (as token ids vLLM would have to re-expand image placeholders it already expanded)
        prompts = [{"prompt": r.prompt + r.text, "multi_modal_data": {"image": list(r.images)}} if r.images
                   else {"prompt_token_ids": r.prompt_token_ids + r.token_ids} for r in results]
        # MiaLLM overwrites out[0].probes with a batch merge that keeps one row per request, so a
        # throwaway first request takes that slot (as in PersVecFast)
        prompts = [prompts[0]] + prompts
        params = [SamplingParams(max_tokens=1, temperature=0.0, extra_args=dict(capture)) for _ in prompts]
        outputs = self.llm.generate(prompts, params)[1:]
        out = []
        for r, o in zip(results, outputs):
            stats = self.llm.analyze(analyzer_spec={"reduce": "none"}, probes=o.probes)["hidden_states"]
            n_prompt, n_total = len(r.prompt_token_ids), len(o.prompt_token_ids)
            per_layer = {}
            for layer in self.layers:
                hs = stats[layer_key(stats.keys(), layer)]
                hs = [hs] if isinstance(hs, torch.Tensor) else hs
                hs = torch.cat([t.reshape(-1, t.shape[-1]).cpu() for t in hs if t is not None])
                if hs.shape[0] != n_total:
                    raise CaptureMismatch(f"layer {layer}: captured {hs.shape[0]} rows for {n_total} prompt+completion "
                                          "tokens; prefix caching or chunked prefill on the capture engine?")
                per_layer[layer] = hs[n_prompt:]
            out.append(per_layer)
        clear_delivered(os.environ["MIA_APERTURE_DIR"])
        return out


@contextmanager
def _visible_gpus(gpus: Optional[str]):
    """Engine processes inherit CUDA_VISIBLE_DEVICES when they are spawned. `gpus` indexes the
    job's own GPUs ("1" = the job's second GPU), so it works under SLURM's renumbering."""
    if gpus is None:
        yield
        return
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    job = old.split(",") if old else [str(i) for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(job[int(i)] for i in str(gpus).split(","))
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def _open_llm(spec: ModelSpec, kind: str, mia_config: dict, work_dir: Path, gpus, memory, max_model_len,
              max_images: int = 0):
    from mia import MiaLLM

    config_file = work_dir / f"{spec.key}_{kind}.json"
    write_json(config_file, mia_config)
    worker, analyzer = {"steer": ("steer", None), "capture": ("capture_hs", "hidden_states")}[kind]
    log(f"loading {spec.hf_repo} ({kind} engine, gpus={gpus or 'all'}, memory={memory})")
    with _visible_gpus(gpus):
        return MiaLLM(model=spec.hf_repo, worker_name=worker, analyzer_name=analyzer, config_file=str(config_file),
                      hook_dir=str(work_dir / f"mia_{kind}"),
                      tensor_parallel_size=spec.tensor_parallel_size, max_model_len=max_model_len,
                      max_num_seqs=spec.max_num_seqs, gpu_memory_utilization=memory,
                      generation_config="vllm",                 # never read the model's generation_config.json
                      enable_prefix_caching=(kind == "steer"),  # a cached prefix would not be re-captured
                      **({"limit_mm_per_prompt": {"image": max_images}} if max_images else {}))


def _shutdown(llm) -> None:
    # MiaLLM.close() is a no-op: stop vLLM's engine process so its GPU memory is freed
    llm.llm.llm_engine.engine_core.shutdown()


def _pixels_per_token(spec: ModelSpec) -> int:
    """Image side length covered by one LM token: vision patch size x spatial merge (32 for Qwen3.5)."""
    from transformers import AutoConfig
    vision = getattr(AutoConfig.from_pretrained(spec.hf_repo), "vision_config", None)
    return getattr(vision, "patch_size", 16) * getattr(vision, "spatial_merge_size", 2) if vision else 32


@contextmanager
def open_engines(model: str, engine_cfg: dict, monitor_layers: Optional[List[int]]):
    """Yield (SteerEngine, CaptureEngine | None). engine_cfg: {max_model_len, sampling, max_images,
    steer: {gpus, memory}, monitor: {gpus, memory}}; no monitor engine when monitor_layers is None."""
    from transformers import AutoTokenizer

    spec = MODELS[model]
    work_dir = Path(os.environ.get("TESTBED_MIA_DIR", "out/_mia")).resolve()
    steer_cfg, mon_cfg = dict(engine_cfg.get("steer") or {}), dict(engine_cfg.get("monitor") or {})
    shared = monitor_layers is not None and steer_cfg.get("gpus") == mon_cfg.get("gpus")
    max_model_len = engine_cfg.get("max_model_len") or spec.max_model_len
    max_images = int(engine_cfg.get("max_images", 0))

    hidden = model_shape(spec)[1]
    zero_vector = work_dir / f"zero_{spec.key}.pt"
    save_pt(zero_vector, {"dir": torch.zeros(hidden)})
    steer_config = {"steering": {"method": "add_vector", "coefficient": 0.0, "optimal_layer": 0,
                                 "vector_path": str(zero_vector)}}

    with ExitStack() as stack:
        llm = _open_llm(spec, "steer", steer_config, work_dir, steer_cfg.get("gpus"),
                        steer_cfg.get("memory", 0.5 if shared else 0.85), max_model_len, max_images)
        stack.callback(_shutdown, llm)
        steer = SteerEngine(llm, spec, AutoTokenizer.from_pretrained(spec.hf_repo), str(zero_vector),
                            dict(engine_cfg.get("sampling") or {}), max_model_len, _pixels_per_token(spec))
        capture = None
        if monitor_layers is not None:
            # capture layers are 1-based: i = output of decoder block i-1
            cap_config = {"hidden_states": {"layers": [l + 1 for l in monitor_layers], "mode": "all_tokens"}}
            cap_llm = _open_llm(spec, "capture", cap_config, work_dir, mon_cfg.get("gpus"),
                                mon_cfg.get("memory", 0.35 if shared else 0.85), max_model_len, max_images)
            stack.callback(_shutdown, cap_llm)
            capture = CaptureEngine(cap_llm, spec, monitor_layers, hidden)
        try:
            yield steer, capture
        finally:
            steer.llm = capture = None
            gc.collect()
