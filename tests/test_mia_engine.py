"""The vLLM-Hook wrappers against a fake MiaLLM: request building, length guard, readout slicing.
Needs PersVecFast importable (pip install -e, or a ../PersVecFast checkout next to this repo)."""
import os
import sys
import types
from pathlib import Path

import pytest
import torch

_pvf_src = Path(__file__).resolve().parents[2] / "PersVecFast" / "src"
if _pvf_src.exists():
    sys.path.insert(0, str(_pvf_src))
pytest.importorskip("persvecfast")


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


sys.modules.setdefault("vllm", types.SimpleNamespace(SamplingParams=_SamplingParams))

from testbed.engine import TurnRequest, TurnResult  # noqa: E402
from testbed.mia_engine import CaptureEngine, CaptureMismatch, SteerEngine  # noqa: E402
from testbed.steering import Steer  # noqa: E402
from persvecfast.models import MODELS  # noqa: E402

SPEC = MODELS["qwen3.5-9b"]


class _Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        def text(content):
            if isinstance(content, str):
                return content
            return " ".join("<img>" if part["type"] == "image" else part["text"] for part in content)
        return " ".join(text(m["content"]) for m in messages)

    def encode(self, text, add_special_tokens):
        return list(range(len(text.split())))


def _out(prompt_ids, token_ids, text="ok", probes=None):
    c = types.SimpleNamespace(text=text, finish_reason="stop", token_ids=token_ids)
    return types.SimpleNamespace(outputs=[c], prompt_token_ids=prompt_ids, probes=probes)


class _SteerLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, params):
        self.calls.append((prompts, params))
        # an image prompt comes back with its placeholders expanded (here: 10 tokens per image)
        return [_out(p.get("prompt_token_ids") or list(range(len(p["prompt"].split()) - 1
                                                             + 10 * len(p["multi_modal_data"]["image"]))), [7, 8])
                for p in prompts]


def test_steer_engine_builds_per_request_configs_and_guards_length():
    llm = _SteerLLM()
    engine = SteerEngine(llm, SPEC, _Tok(), "/zero.pt", {"temperature": 0.3}, max_model_len=40)
    steer = Steer(layer=12, coeff=2.5, vector_path="/v.pt", label="x")
    requests = [TurnRequest(("a",), "sys", "short question", seed=1, steer=steer, max_tokens=100),
                TurnRequest(("b",), "sys", "short question", seed=2, max_tokens=10),
                TurnRequest(("c",), "sys", "word " * 60, seed=3, max_tokens=10)]
    results = engine.generate(requests)
    prompts, params = llm.calls[0]
    assert len(prompts) == 2  # the over-long prompt never reaches vLLM
    assert params[0].extra_args["steer"] == {"method": "add_vector", "coefficient": 2.5,
                                             "optimal_layer": 12, "vector_path": "/v.pt"}
    assert params[1].extra_args["steer"]["coefficient"] == 0.0
    assert params[1].extra_args["steer"]["vector_path"] == "/zero.pt"
    assert params[0].max_tokens == 40 - 3 - 1 and params[1].max_tokens == 10  # clamped to the room left
    assert params[0].temperature == 0.3 and params[0].seed == 1  # sampling override, own seed
    assert [r.key for r in results] == [("a",), ("b",), ("c",)]
    assert results[2].finish_reason == "prompt_too_long" and results[2].text == ""
    assert results[0].token_ids == [7, 8] and results[0].prompt_token_ids == [0, 1, 2]


class _CaptureLLM:
    def __init__(self, extra_rows=0):
        self.extra_rows = extra_rows
        self.prompts = None

    def generate(self, prompts, params):
        self.prompts = prompts
        return [_out(p["prompt_token_ids"], [0], probes=len(p["prompt_token_ids"])) for p in prompts]

    def analyze(self, analyzer_spec, probes):
        n = probes + self.extra_rows
        rows = torch.arange(n, dtype=torch.float).unsqueeze(1).repeat(1, 4)  # row i == position i
        return {"hidden_states": {"model.layers.3": rows, "model.layers.5": [rows[:2], rows[2:]]}}


def test_capture_engine_keeps_completion_rows(monkeypatch):
    monkeypatch.setenv("MIA_APERTURE_DIR", "unused")
    llm = _CaptureLLM()
    engine = CaptureEngine(llm, SPEC, layers=[3, 5], hidden=4)
    results = [TurnResult(("a",), "x", "stop", [1, 2, 3], [4, 5]), TurnResult(("b",), "", "stop", [1], []),
               TurnResult(("c",), "y", "stop", [1, 2], [3, 4, 5])]
    rows = engine.read(results)
    # a throwaway first request takes the slot MiaLLM's batch merge overwrites
    assert llm.prompts[0] == llm.prompts[1] == {"prompt_token_ids": [1, 2, 3, 4, 5]}
    assert rows[0][3][:, 0].tolist() == [3.0, 4.0] and rows[0][5].shape == (2, 4)
    assert rows[1][3].shape == (0, 4)  # empty completion: not sent, empty rows
    assert rows[2][5][:, 0].tolist() == [2.0, 3.0, 4.0]


def test_capture_row_count_is_checked(monkeypatch):
    monkeypatch.setenv("MIA_APERTURE_DIR", "unused")
    engine = CaptureEngine(_CaptureLLM(extra_rows=1), SPEC, layers=[3], hidden=4)
    with pytest.raises(CaptureMismatch):
        engine.read([TurnResult(("a",), "x", "stop", [1, 2], [3])])


def test_image_requests_go_in_as_text_with_multimodal_data():
    from PIL import Image
    llm = _SteerLLM()
    engine = SteerEngine(llm, SPEC, _Tok(), "/zero.pt", {}, max_model_len=200, pixels_per_token=32)
    img = Image.new("RGB", (64, 96))  # 2 x 3 tokens + 2 markers = 8
    res = engine.generate([TurnRequest(("a",), "sys", "what now", seed=1, images=[img], max_tokens=500)])[0]
    prompt, = llm.calls[0][0]
    assert prompt["prompt"] == "sys <img> what now" and prompt["multi_modal_data"] == {"image": [img]}
    assert llm.calls[0][1][0].max_tokens == 200 - 4 - 8 - 1  # text tokens and image tokens both count
    assert res.prompt == "sys <img> what now" and res.images == [img]
    huge = Image.new("RGB", (640, 640))  # 400 tokens: no room left
    assert engine.generate([TurnRequest(("b",), "s", "u", seed=1, images=[huge])])[0].finish_reason == "prompt_too_long"


def test_capture_rereads_image_turns_from_text_and_images(monkeypatch):
    from PIL import Image
    monkeypatch.setenv("MIA_APERTURE_DIR", "unused")

    class _ImgCaptureLLM(_CaptureLLM):
        def generate(self, prompts, params):
            self.prompts = prompts
            n = [len(p["prompt"].split()) - 1 + 10 for p in prompts]  # expanded: 10 tokens per image
            return [_out(list(range(k)), [0], probes=k) for k in n]

    llm = _ImgCaptureLLM()
    img = Image.new("RGB", (8, 8))
    # the generation saw an expanded prompt of 12 tokens and produced 3 tokens
    turn = TurnResult(("a",), "go left now", "stop", list(range(12)), [5, 6, 7], prompt="sys <img> u ", images=[img])
    rows = CaptureEngine(llm, SPEC, layers=[3], hidden=4).read([turn])
    assert llm.prompts[1] == {"prompt": "sys <img> u go left now", "multi_modal_data": {"image": [img]}}
    assert rows[0][3][:, 0].tolist() == [12.0, 13.0, 14.0]  # everything after the 12 prompt rows


def test_prompt_only_steer_sends_zero_vector():
    llm = _SteerLLM()
    engine = SteerEngine(llm, SPEC, _Tok(), "/zero.pt", {}, max_model_len=100)
    persona = Steer(layer=None, coeff=0.0, vector_path=None, label="prompt(x)", persona="Be x.")
    engine.generate([TurnRequest(("a",), "sys", "q", seed=1, steer=persona)])
    assert llm.calls[0][1][0].extra_args["steer"]["vector_path"] == "/zero.pt"
    assert llm.calls[0][1][0].extra_args["steer"]["coefficient"] == 0.0


class DeliveryReadTimeout(Exception):
    """Stands in for mia's exception of the same name (matched by name)."""


def test_capture_retries_a_lost_delivery_then_skips(tmp_path, monkeypatch):
    import testbed.mia_engine as me
    monkeypatch.setenv("MIA_APERTURE_DIR", str(tmp_path))
    monkeypatch.setattr(me, "clear_delivered", lambda d: None)
    eng = CaptureEngine(llm=None, spec=SPEC, layers=[2], hidden=4)
    turns = [TurnResult(("r", 0, "a", 0), "hi", "stop", [1, 2], [3], "x", [])]
    calls = []

    def flaky(results):
        calls.append(1)
        if len(calls) == 1:
            raise DeliveryReadTimeout("1 of 1 request(s) not delivered")
        return [{2: torch.ones(1, 4)} for _ in results]

    monkeypatch.setattr(eng, "_read", flaky)
    assert eng.read(turns)[0][2].shape == (1, 4) and len(calls) == 2   # retried, then delivered

    monkeypatch.setattr(eng, "_read", lambda results: (_ for _ in ()).throw(DeliveryReadTimeout("lost")))
    assert eng.read(turns)[0][2].shape == (0, 4)                        # lost twice: no readout, no crash

    monkeypatch.setattr(eng, "_read", lambda results: (_ for _ in ()).throw(ValueError("real bug")))
    with pytest.raises(ValueError):
        eng.read(turns)                                                 # other errors still surface
