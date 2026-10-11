"""Tests for the cls system's inference model."""

import pytest
import torch
from torch import nn

from espnet3.systems.cls.inference_model import ClsInferenceModel
from espnet3.systems.cls.model import ClsModel

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_loads_a_plain_state_dict               | Averaged *.pth weights load  |
# |                          | even though they carry the loss weights.        |
# | test_loads_a_lightning_checkpoint           | Per-epoch checkpoints keep   |
# |                                             | weights under state_dict.    |
# | test_labels_one_waveform                    | Returns (index, probs, name) |
# |                                             | in token_list order.         |


class TinyFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 4)

    def output_size(self):
        return 4

    def forward(self, speech, speech_lengths):
        frames = speech[:, : (speech.size(1) // 160) * 160]
        frames = frames.reshape(speech.size(0), -1, 160).mean(-1, keepdim=True)
        return self.projection(frames), torch.div(
            speech_lengths, 160, rounding_mode="floor"
        )


def _network(**kwargs):
    torch.manual_seed(0)
    return ClsModel(TinyFrontend(), num_classes=3, head_size=8, **kwargs)


@pytest.fixture
def token_list(tmp_path):
    path = tmp_path / "token_list"
    path.write_text("angry\nhappy\nsad\n<unk>\n")
    return path


def _trained_state():
    return _network(class_weights=[1.0, 2.0, 3.0]).state_dict()


def test_loads_a_plain_state_dict(tmp_path, token_list):
    path = tmp_path / "model.pth"
    state = _trained_state()
    torch.save(state, path)

    model = ClsInferenceModel(_network(), path, token_list)

    assert torch.equal(model.model.head.output.weight, state["head.output.weight"])


def test_loads_a_lightning_checkpoint(tmp_path, token_list):
    path = tmp_path / "epoch000.ckpt"
    torch.save({"epoch": 0, "state_dict": _trained_state()}, path)

    model = ClsInferenceModel(_network(), path, token_list)

    assert not model.model.training


def test_labels_one_waveform(tmp_path, token_list):
    path = tmp_path / "model.pth"
    torch.save(_trained_state(), path)
    model = ClsInferenceModel(_network(), path, token_list)

    prediction, scores, label = model(torch.randn(16000).numpy())

    assert scores.shape == (3,)
    assert float(scores.sum()) == pytest.approx(1.0)
    assert label == ["angry", "happy", "sad"][prediction[0]]
