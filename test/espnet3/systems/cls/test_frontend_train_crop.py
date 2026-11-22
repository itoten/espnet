"""Tests for ``Wav2Vec2Frontend``'s ``max_train_crop``."""

import pytest
import torch
from torch import nn

from espnet3.systems.cls.src.frontend import Wav2Vec2Frontend

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_crops_a_batch_with_little_padding      | Cut to the shortest waveform |
# |                          | when that removes at most the allowed fraction; |
# |                          | the attention mask is then all ones.            |
# | test_keeps_a_batch_with_much_padding        | Over the fraction: no cut.   |
# | test_never_crops_in_eval                    | Evaluation sees every sample.|
# | test_zero_never_crops                       | The default leaves batches   |
# |                                             | as they are.                 |

HIDDEN = 8


class _Config:
    hidden_size = HIDDEN


class _FeatureEncoder(nn.Module):
    def _freeze_parameters(self):
        pass


class FakeEncoder(nn.Module):
    """Records what it was given; one frame per 320 samples."""

    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.feature_extractor = _FeatureEncoder()
        self.projection = nn.Linear(1, HIDDEN)
        self.seen = {}

    def _get_feat_extract_output_lengths(self, input_lengths):
        return torch.div(input_lengths, 320, rounding_mode="floor")

    def forward(self, speech, attention_mask=None):
        self.seen["speech"] = speech
        self.seen["attention_mask"] = attention_mask
        frames = speech.size(1) // 320
        hidden = self.projection(speech[:, :frames].unsqueeze(-1))
        return type("Output", (), {"last_hidden_state": hidden})()


def _run(lengths, max_train_crop, training=True, total=16000):
    encoder = FakeEncoder()
    frontend = Wav2Vec2Frontend(encoder, max_train_crop=max_train_crop)
    frontend.train(training)
    speech = torch.randn(len(lengths), total)
    _, frame_lengths = frontend(speech, torch.tensor(lengths))
    return encoder.seen, frame_lengths


def test_crops_a_batch_with_little_padding():
    seen, frame_lengths = _run([16000, 15200], max_train_crop=0.1)

    assert seen["speech"].shape == (2, 15200)
    assert bool(seen["attention_mask"].all())
    assert frame_lengths.tolist() == [15200 // 320] * 2


def test_keeps_a_batch_with_much_padding():
    seen, frame_lengths = _run([16000, 8000], max_train_crop=0.1)

    assert seen["speech"].shape == (2, 16000)
    assert frame_lengths.tolist() == [16000 // 320, 8000 // 320]


def test_never_crops_in_eval():
    seen, _ = _run([16000, 15200], max_train_crop=0.1, training=False)

    assert seen["speech"].shape == (2, 16000)


@pytest.mark.parametrize("lengths", [[16000, 15999], [16000, 16000]])
def test_zero_never_crops(lengths):
    seen, _ = _run(lengths, max_train_crop=0.0)

    assert seen["speech"].shape == (2, 16000)
