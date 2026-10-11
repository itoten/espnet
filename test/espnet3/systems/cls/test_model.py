"""Tests for the cls classifier."""

import numpy as np
import pytest
import torch
from torch import nn

from espnet3.systems.cls.model import (
    ClassificationHead,
    ClsModel,
    WaveformNormalize,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# WaveformNormalize
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_normalize_matches_the_collected_statistics | The mean and deviation   |
# |                          | come from what collect_stats wrote.             |
# | test_normalize_standardises_the_waveform    | Output has zero mean and     |
# |                                             | unit deviation.              |
# | test_normalize_does_not_allocate_a_length_square | A waveform is too long  |
# |                          | for GlobalMVN's padding mask.                   |
#
# ClassificationHead
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_head_maps_pooled_vector_to_logits      | Output width is the class    |
# |                                             | count.                       |
# | test_head_stacks_the_requested_blocks       | num_layers controls the      |
# |                                             | hidden block count.          |
# | test_head_rejects_zero_layers               | A head needs one block.      |
#
# ClsModel
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_returns_the_espnet3_triple     | (loss, stats, weight), with  |
# |                          | a differentiable loss and detached stats.       |
# | test_pools_only_the_frames_the_frontend_reports | A shorter waveform  |
# |                          | pools over fewer frames, not over padding.      |
# | test_frontend_starts_in_train_mode          | from_pretrained returns eval |
# |                          | mode, and Lightning does not correct it.         |
# | test_compute_logits_maps_waveforms_to_classes | Inference entry point. |
# | test_class_weights_reach_the_loss           | Weighting a class changes    |
# |                                             | the loss.                    |
# | test_class_weights_can_come_from_a_file     | A path is read by the model, |
# |                          | since prepare_labels writes it after config load.|
# | test_rejects_mismatched_class_weights       | A wrong weight count is an   |
# |                                             | error.                       |
# | test_rejects_multi_label_batches            | The model is single-label.   |
# | test_collect_feats_exposes_the_waveform     | collect_stats measures the   |
# |                                             | waveform it returns.         |

FRAME_SIZE = 16
SAMPLES_PER_FRAME = 320


class FakeFrontend(nn.Module):
    """Stands in for a frontend: one frame per 320 samples."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, FRAME_SIZE)
        self.seen = {}

    def output_size(self):
        return FRAME_SIZE

    def forward(self, speech, speech_lengths):
        frame_lengths = torch.div(
            speech_lengths, SAMPLES_PER_FRAME, rounding_mode="floor"
        )
        frames = speech[:, : int(frame_lengths.max()) * SAMPLES_PER_FRAME]
        frames = frames.reshape(speech.size(0), -1, SAMPLES_PER_FRAME).mean(-1)
        self.seen["frame_lengths"] = frame_lengths
        return self.projection(frames.unsqueeze(-1)), frame_lengths


def _batch(batch_size=3, samples=16000):
    return {
        "speech": torch.randn(batch_size, samples),
        "speech_lengths": torch.full((batch_size,), samples, dtype=torch.long),
        "label": torch.arange(batch_size).unsqueeze(-1) % 7,
        "label_lengths": torch.ones(batch_size, dtype=torch.long),
    }


def _model(**kwargs):
    kwargs.setdefault("num_classes", 7)
    kwargs.setdefault("head_size", 8)
    return ClsModel(FakeFrontend(), **kwargs)


def _stats_file(tmp_path, values):
    """Write the statistics collect_stats produces for one waveform set."""
    path = tmp_path / "feats_stats.npz"
    array = np.asarray(values, dtype=np.float64)
    np.savez(
        path,
        count=np.array(array.size),
        sum=np.array(array.sum()),
        sum_square=np.array((array**2).sum()),
    )
    return path


def test_normalize_matches_the_collected_statistics(tmp_path):
    values = np.array([1.0, 2.0, 3.0, 4.0])
    normalize = WaveformNormalize(_stats_file(tmp_path, values), eps=0.0)

    assert float(normalize.mean) == pytest.approx(values.mean())
    assert float(normalize.std) == pytest.approx(values.std())


def test_normalize_standardises_the_waveform(tmp_path):
    values = np.array([1.0, 2.0, 3.0, 4.0])
    normalize = WaveformNormalize(_stats_file(tmp_path, values), eps=0.0)

    speech = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    output, lengths = normalize(speech, torch.tensor([4]))

    assert float(output.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(output.std(unbiased=False)) == pytest.approx(1.0, abs=1e-5)
    assert lengths.tolist() == [4]


def test_normalize_does_not_allocate_a_length_square(tmp_path):
    """A waveform is too long for a padding mask.

    GlobalMVN masks padding with a (T, T) matrix, which 144000 samples cannot
    afford: it would ask for 20 GB.
    """
    normalize = WaveformNormalize(_stats_file(tmp_path, np.random.randn(1000)))

    output, _ = normalize(torch.randn(2, 144000), torch.tensor([144000, 70000]))

    assert output.shape == (2, 144000)


def test_head_maps_pooled_vector_to_logits():
    head = ClassificationHead(32, 8, 7)
    assert head(torch.randn(2, 32)).shape == (2, 7)


def test_head_stacks_the_requested_blocks():
    assert len(ClassificationHead(32, 8, 7, num_layers=3).blocks) == 3


def test_head_rejects_zero_layers():
    with pytest.raises(ValueError, match="at least 1"):
        ClassificationHead(32, 8, 7, num_layers=0)


def test_forward_returns_the_espnet3_triple():
    model = _model()
    loss, stats, weight = model(**_batch())

    assert loss.ndim == 0 and loss.requires_grad
    assert set(stats) == {"loss", "acc"}
    assert not any(value.requires_grad for value in stats.values())
    assert float(weight) == 3.0


def test_pools_only_the_frames_the_frontend_reports():
    """A short utterance batched with a long one gets the same logits alone."""
    model = _model().eval()
    speech = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 8000])

    with torch.no_grad():
        batched = model.compute_logits(speech, lengths)[1]
        alone = model.compute_logits(speech[1:, :8000], lengths[1:])[0]

    assert torch.allclose(batched, alone, atol=1e-6)


def test_frontend_starts_in_train_mode():
    """The frontend must not start in eval mode.

    Lightning never switches modes at fit start, and from_pretrained hands back
    an encoder in eval mode, so its dropout would stay off for the whole run.
    """
    frontend = FakeFrontend()
    frontend.eval()

    model = ClsModel(frontend, num_classes=7, head_size=8)

    assert model.training
    assert model.frontend.training


def test_compute_logits_maps_waveforms_to_classes():
    model = _model()
    batch = _batch(batch_size=2)

    logits = model.compute_logits(batch["speech"], batch["speech_lengths"])

    assert logits.shape == (2, 7)


def test_class_weights_reach_the_loss():
    """Two classes are needed: weighting one class alone cancels in the mean."""
    batch = _batch(batch_size=2)
    batch["label"] = torch.tensor([[0], [1]])

    torch.manual_seed(0)
    flat, _, _ = _model(class_weights=[1.0] * 7)(**batch)
    torch.manual_seed(0)
    leaning, _, _ = _model(class_weights=[5.0] + [1.0] * 6)(**batch)

    assert not torch.isclose(flat, leaning)


def test_class_weights_can_come_from_a_file(tmp_path):
    """The model reads the path itself.

    prepare_labels writes the file after the config is loaded.
    """
    weights_file = tmp_path / "class_weights"
    weights_file.write_text("1.5\n0.5\n2.0\n")

    model = _model(num_classes=3, class_weights=weights_file)

    assert model.class_weights.tolist() == [1.5, 0.5, 2.0]


def test_rejects_mismatched_class_weights():
    with pytest.raises(ValueError, match="holds 3 values"):
        _model(class_weights=[1.0, 1.0, 1.0])


def test_rejects_multi_label_batches():
    model = _model()
    batch = _batch(batch_size=2)
    batch["label_lengths"] = torch.tensor([1, 2])

    with pytest.raises(ValueError, match="single-label"):
        model(**batch)


def test_collect_feats_exposes_the_waveform():
    model = _model()
    batch = _batch()

    feats = model.collect_feats(**batch)

    assert torch.equal(feats["feats"], batch["speech"])
    assert torch.equal(feats["feats_lengths"], batch["speech_lengths"])
