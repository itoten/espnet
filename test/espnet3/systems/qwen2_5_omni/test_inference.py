"""Tests for Qwen2_5OmniInference, with the transformers classes stubbed."""

import numpy as np
import pytest
import torch

import espnet3.systems.qwen2_5_omni.inference as module
from espnet3.systems.qwen2_5_omni.inference import Qwen2_5OmniInference

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                  | Description                   |
# |--------------------------------------------|-------------------------------|
# | test_loads_without_talker                  | enable_audio_output=False,    |
# |                                            | talker disabled, dtype set.   |
# | test_conversation_puts_audio_before_prompt | One user turn: audio, text.   |
# | test_system_prompt_is_optional             | None keeps the template's.    |
# | test_generate_is_greedy_and_strips_prompt  | Thinker, do_sample=False, max |
# |                                            | tokens, answer only.          |
# | test_call_accepts_one_or_a_batch           | str in, str out; lists too.   |
# | test_generate_rejects_mismatched_lengths   | One prompt per clip.          |

PROMPT_LEN = 3


class FakeThinker:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        batch = kwargs["input_ids"].shape[0]
        answer = torch.arange(100, 102).repeat(batch, 1)
        return torch.cat([kwargs["input_ids"], answer], dim=1)


class FakeModel:
    def __init__(self):
        self.talker_disabled = False
        self.thinker = FakeThinker()

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        model = cls()
        model.load_args = (name, kwargs)
        return model

    def disable_talker(self):
        self.talker_disabled = True

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        raise AssertionError("text output must call the thinker directly")


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    @classmethod
    def from_pretrained(cls, name):
        processor = cls()
        processor.calls = []
        return processor

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return repr(messages)

    def __call__(self, text, audio, sampling_rate, return_tensors, padding):
        self.calls.append(dict(text=text, audio=audio, sampling_rate=sampling_rate))
        batch = len(text)
        return FakeInputs(
            input_ids=torch.ones(batch, PROMPT_LEN, dtype=torch.long),
            input_features=torch.zeros(batch, 128, 10),
        )

    def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        return [" ".join(str(int(i)) for i in row) for row in ids]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(module, "Qwen2_5OmniForConditionalGeneration", FakeModel)
    monkeypatch.setattr(module, "Qwen2_5OmniProcessor", FakeProcessor)
    return Qwen2_5OmniInference(model_name="some/omni", max_new_tokens=7)


def test_loads_without_talker(model):
    name, kwargs = model.model.load_args
    assert name == "some/omni"
    assert kwargs["enable_audio_output"] is False
    assert kwargs["dtype"] is torch.bfloat16
    assert model.model.talker_disabled


def test_conversation_puts_audio_before_prompt(model):
    messages = model.build_conversation("Which emotion?")

    assert messages == [
        {
            "role": "user",
            "content": [{"type": "audio"}, {"type": "text", "text": "Which emotion?"}],
        }
    ]


def test_system_prompt_is_optional(model):
    model.system_prompt = "You are a speech emotion recognition model."

    messages = model.build_conversation("x")

    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["text"] == model.system_prompt


def test_generate_is_greedy_and_strips_prompt(model):
    answers = model.generate([np.zeros(160), torch.zeros(320)], ["p1", "p2"])

    assert answers == ["100 101", "100 101"]
    kwargs = model.model.thinker.generate_kwargs
    assert kwargs["do_sample"] is False
    assert kwargs["max_new_tokens"] == 7
    assert kwargs["input_features"].dtype is torch.bfloat16
    call = model.processor.calls[0]
    assert call["sampling_rate"] == 16000
    assert [a.dtype for a in call["audio"]] == [np.float32, np.float32]


def test_call_accepts_one_or_a_batch(model):
    assert model(speech=np.zeros(160), prompt="p") == "100 101"
    assert model(speech=[np.zeros(160)] * 2, prompt=["p", "q"]) == ["100 101"] * 2


def test_generate_rejects_mismatched_lengths(model):
    with pytest.raises(ValueError, match="2 waveforms for 1 prompts"):
        model.generate([np.zeros(1), np.zeros(1)], ["p"])
