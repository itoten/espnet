"""Tests for attentive statistics pooling."""

import pytest
import torch

from espnet3.systems.wavlmasp.pooling import AttentiveStatisticsPooling

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_output_is_twice_the_input_width        | A mean and a deviation are   |
# |                                             | concatenated.                |
# | test_padding_does_not_reach_the_output      | Frames past a sequence's     |
# |                          | length leave the pooled vector unchanged.       |
# | test_matches_a_hand_computed_weighted_mean  | Uniform attention reduces to |
# |                          | the plain mean and deviation.                   |
# | test_rejects_lengths_past_the_padding       | A frame count larger than    |
# |                          | the batch is an error, not a silent clamp.      |
# | test_deviation_stays_finite_on_one_frame    | A single frame has zero      |
# |                          | variance, which must not produce a nan.         |


def test_output_is_twice_the_input_width():
    pool = AttentiveStatisticsPooling(4)
    pooled = pool(torch.randn(3, 10, 4), torch.tensor([10, 7, 1]))

    assert pooled.shape == (3, 8)
    assert pool.output_size == 8


def test_padding_does_not_reach_the_output():
    pool = AttentiveStatisticsPooling(4)
    frames = torch.randn(2, 6, 4)
    lengths = torch.tensor([6, 3])

    padded = frames.clone()
    padded[1, 3:] = 1e3  # garbage past the second sequence's length

    assert torch.allclose(pool(frames, lengths), pool(padded, lengths))


def test_matches_a_hand_computed_weighted_mean():
    """With the attention zeroed the softmax is uniform, so plain statistics."""
    pool = AttentiveStatisticsPooling(2)
    with torch.no_grad():
        pool.attention.zero_()

    frames = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    pooled = pool(frames, torch.tensor([3]))

    expected_mean = frames.mean(dim=1)
    expected_dev = frames.var(dim=1, unbiased=False).sqrt()
    assert torch.allclose(pooled[:, :2], expected_mean, atol=1e-5)
    assert torch.allclose(pooled[:, 2:], expected_dev, atol=1e-4)


def test_rejects_lengths_past_the_padding():
    pool = AttentiveStatisticsPooling(4)
    with pytest.raises(ValueError, match="must come from the encoder"):
        pool(torch.randn(2, 5, 4), torch.tensor([6, 5]))


def test_deviation_stays_finite_on_one_frame():
    pool = AttentiveStatisticsPooling(4)
    pooled = pool(torch.randn(1, 5, 4), torch.tensor([1]))

    assert torch.isfinite(pooled).all()
