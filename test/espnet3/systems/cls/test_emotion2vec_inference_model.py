"""Tests for the cls system's zero-shot emotion2vec+ inference model."""

import sys
import types

import numpy as np
import pytest
import torch

from espnet3.systems.cls.src.emotion2vec_inference_model import (
    Emotion2vecInferenceModel,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                  | Description                   |
# |--------------------------------------------|-------------------------------|
# | test_maps_and_renormalises_scores          | Mapped labels are gathered in |
# |                       | token_list order; other / <unk> are dropped.     |
# | test_argmax_ignores_unmapped_labels        | A dominant `other` cannot be  |
# |                                            | predicted.                    |
# | test_unmapped_class_scores_zero            | A class with no source label  |
# |                                            | gets 0 and a warning.         |
# | test_rejects_unknown_model_label           | Typos in label_map fail early.|
# | test_rejects_unknown_class                 | Classes outside token_list    |
# |                                            | fail early.                   |
# | test_rejects_two_labels_for_one_class      | Ambiguous maps fail early.    |
# | test_accepts_a_tensor                      | Waveform tensors are accepted.|

MODEL_TOKENS = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
]

LABEL_MAP = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprise",
}


class FakeAutoModel:
    """Stands in for ``funasr.AutoModel``; returns preset scores."""

    scores = None
    calls = []

    def __init__(self, model, hub, device, disable_update):
        self.kwargs = {"token_list": list(MODEL_TOKENS)}
        FakeAutoModel.init_args = dict(model=model, hub=hub, device=device)

    def generate(self, waveform, **kwargs):
        FakeAutoModel.calls.append((waveform, kwargs))
        return [{"labels": list(MODEL_TOKENS), "scores": list(self.scores)}]


@pytest.fixture(autouse=True)
def fake_funasr(monkeypatch):
    module = types.ModuleType("funasr")
    module.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "funasr", module)
    FakeAutoModel.calls = []
    yield


@pytest.fixture
def token_list(tmp_path):
    path = tmp_path / "token_list"
    path.write_text("angry\nhappy\nsad\nfear\nsurprise\ndisgust\nneutral\n<unk>\n")
    return path


def _model(token_list, label_map=LABEL_MAP, scores=None):
    FakeAutoModel.scores = scores or [0.1] * len(MODEL_TOKENS)
    return Emotion2vecInferenceModel(token_list=token_list, label_map=label_map)


def test_maps_and_renormalises_scores(token_list):
    # angry disgusted fearful happy neutral other sad surprised <unk>
    raw = [0.30, 0.02, 0.03, 0.10, 0.20, 0.20, 0.05, 0.05, 0.05]
    model = _model(token_list, scores=raw)

    prediction, scores, label = model(np.zeros(16000, dtype=np.float32))

    # token_list order: angry happy sad fear surprise disgust neutral
    expected = torch.tensor([0.30, 0.10, 0.05, 0.03, 0.05, 0.02, 0.20])
    expected = expected / expected.sum()
    assert torch.allclose(scores, expected)
    assert prediction == [0]
    assert label == "angry"
    assert FakeAutoModel.calls[0][1]["granularity"] == "utterance"
    assert FakeAutoModel.init_args["model"] == "emotion2vec/emotion2vec_plus_large"


def test_argmax_ignores_unmapped_labels(token_list):
    raw = [0.01, 0.01, 0.01, 0.01, 0.10, 0.80, 0.02, 0.02, 0.02]
    _, scores, label = _model(token_list, scores=raw)(np.zeros(160))

    assert label == "neutral"
    assert scores.sum().item() == pytest.approx(1.0)


def test_unmapped_class_scores_zero(token_list, caplog):
    label_map = {k: v for k, v in LABEL_MAP.items() if k != "disgusted"}
    model = _model(token_list, label_map=label_map)

    _, scores, _ = model(np.zeros(160))

    assert scores[5].item() == 0.0  # disgust
    assert "disgust" in caplog.text


def test_rejects_unknown_model_label(token_list):
    with pytest.raises(ValueError, match="labels the model does not have"):
        _model(token_list, label_map={"anger": "angry"})


def test_rejects_unknown_class(token_list):
    with pytest.raises(ValueError, match="classes not in token_list"):
        _model(token_list, label_map={"angry": "anger"})


def test_rejects_two_labels_for_one_class(token_list):
    with pytest.raises(ValueError, match="more than one label"):
        _model(token_list, label_map={"angry": "angry", "other": "angry"})


def test_accepts_a_tensor(token_list):
    model = _model(token_list)

    model(torch.zeros(160))

    waveform = FakeAutoModel.calls[0][0]
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.float32
