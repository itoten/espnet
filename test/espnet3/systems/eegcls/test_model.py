"""Tests for the eegcls model."""

import pytest
import torch
from torch import nn

from espnet3.systems.eegcls.model import EegClsModel

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_compute_logits_gives_channels_first    | (B, T, C) reaches the network|
# |                                             | as (B, C, T).                |
# | test_forward_returns_the_espnet3_triple     | (loss, stats, weight) with a |
# |                          | differentiable loss and detached stats.         |
# | test_forward_loss_matches_cross_entropy     | The loss is plain CE.        |
# | test_rejects_multi_label_batches            | The model is single-label.   |
# | test_rejects_flat_labels                    | label must be (B, 1).        |
# | test_collect_feats_exposes_the_epochs       | collect_stats measures the   |
# |                                             | epochs it returns.           |


class _Probe(nn.Module):
    """Network that records its input and returns fixed-size logits."""

    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.linear = nn.Linear(channels, classes)
        self.seen = None

    def forward(self, x):
        self.seen = x
        return self.linear(x.mean(dim=-1))


def _batch(batch=4, time=20, channels=3):
    eeg = torch.randn(batch, time, channels)
    lengths = torch.full((batch,), time)
    label = torch.tensor([[0], [1], [1], [0]][:batch])
    label_lengths = torch.ones(batch, dtype=torch.long)
    return eeg, lengths, label, label_lengths


def test_compute_logits_gives_channels_first():
    network = _Probe(channels=3, classes=2)
    model = EegClsModel(network)
    eeg = torch.randn(4, 20, 3)

    logits = model.compute_logits(eeg)

    assert network.seen.shape == (4, 3, 20)
    assert logits.shape == (4, 2)


def test_forward_returns_the_espnet3_triple():
    model = EegClsModel(_Probe(3, 2))

    loss, stats, weight = model(*_batch())

    assert loss.requires_grad
    assert set(stats) == {"loss", "acc"}
    assert not stats["loss"].requires_grad
    assert weight.item() == 4


def test_forward_loss_matches_cross_entropy():
    model = EegClsModel(_Probe(3, 2))
    eeg, lengths, label, label_lengths = _batch()

    loss, _, _ = model(eeg, lengths, label, label_lengths)

    expected = torch.nn.functional.cross_entropy(model.compute_logits(eeg), label[:, 0])
    assert loss.item() == pytest.approx(expected.item())


def test_rejects_multi_label_batches():
    model = EegClsModel(_Probe(3, 2))
    eeg, lengths, label, _ = _batch()

    with pytest.raises(ValueError, match="single-label"):
        model(eeg, lengths, label, torch.full((4,), 2))


def test_rejects_flat_labels():
    model = EegClsModel(_Probe(3, 2))
    eeg, lengths, label, label_lengths = _batch()

    with pytest.raises(ValueError, match="label must be"):
        model(eeg, lengths, label[:, 0], label_lengths)


def test_collect_feats_exposes_the_epochs():
    model = EegClsModel(_Probe(3, 2))
    eeg = torch.randn(4, 20, 3)
    lengths = torch.full((4,), 20)

    feats = model.collect_feats(eeg, lengths)

    assert feats["feats"] is eeg
    assert feats["feats_lengths"] is lengths
