"""Verify that the CAA prompt generators produce obs dicts compatible with renderers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from scripts.extract_steering_vector import _beauty_contest_prompts, _gbs_prompts
from testbed.registry import build_game


def test_beauty_contest_prompts_render_without_error():
    _, renderer, _ = build_game(
        family="symbolic", game_id="beauty_contest", num_players=4, env_kwargs={})
    pairs = _beauty_contest_prompts(renderer, num_samples=8)
    assert len(pairs) == 8
    for system, user in pairs:
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and "CHOICE:" in user


def test_gbs_prompts_render_without_error():
    _, renderer, _ = build_game(
        family="symbolic", game_id="gbs", num_players=4, env_kwargs={})
    pairs = _gbs_prompts(renderer, num_samples=8)
    assert len(pairs) == 8
    for system, user in pairs:
        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and "GUESS:" in user


class _FakeBatch(dict):
    def to(self, device):
        return self


class _RecordingTokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        return "templated"

    def __call__(self, text, return_tensors=None):
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})


def test_last_token_hidden_disables_thinking_mode():
    from scripts.extract_steering_vector import _last_token_hidden

    class _TinyLayer(nn.Module):
        def forward(self, x):
            return x

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _TinyLayer()

        def forward(self, input_ids):
            x = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
            return self.layer(x)

    model = _TinyModel()
    tok = _RecordingTokenizer()
    _last_token_hidden(model, tok, "sys", "user", model.layer, "cpu")
    assert tok.template_calls[-1]["enable_thinking"] is False
