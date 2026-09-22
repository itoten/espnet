"""Attentive statistics pooling over self-supervised frame representations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["AttentiveStatisticsPooling"]


class AttentiveStatisticsPooling(nn.Module):
    """Pool a frame sequence into a weighted mean and standard deviation.

    Each frame is scored by a learned attention vector, the scores are softmaxed
    over the sequence, and the weighted mean and standard deviation are
    concatenated. The output therefore has twice the input width.

    Introduced by Okabe et al., "Attentive Statistics Pooling for Deep Speaker
    Embedding" (Interspeech 2018).

    Args:
        input_size: Width of one frame.

    Examples:
        >>> pool = AttentiveStatisticsPooling(4)
        >>> frames = torch.randn(2, 10, 4)
        >>> lengths = torch.tensor([10, 6])
        >>> pool(frames, lengths).shape
        torch.Size([2, 8])
    """

    def __init__(self, input_size: int) -> None:
        """Initialize the attention projection and its query vector."""
        super().__init__()
        self.input_size = int(input_size)
        self.sap_linear = nn.Linear(self.input_size, self.input_size)
        self.attention = nn.Parameter(torch.FloatTensor(self.input_size, 1))
        nn.init.normal_(self.attention, mean=0.0, std=1.0)

    @property
    def output_size(self) -> int:
        """Width of the pooled vector, which holds a mean and a deviation."""
        return self.input_size * 2

    def forward(self, frames: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pool every sequence over its own valid frames.

        Args:
            frames: Frame representations of shape ``(B, T, input_size)``.
            lengths: Valid frame count per sequence, shape ``(B,)``. Padding
                beyond a sequence's length takes no part in the attention.

        Returns:
            torch.Tensor: Pooled vectors of shape ``(B, input_size * 2)``.

        Raises:
            ValueError: If a length exceeds the padded sequence.
        """
        max_frames = frames.size(1)
        if int(lengths.max()) > max_frames:
            raise ValueError(
                f"lengths hold {int(lengths.max())} frames but the batch is padded "
                f"to {max_frames}. The frame count must come from the encoder that "
                "produced these frames."
            )

        pooled = []
        for sequence, length in zip(frames, lengths.tolist()):
            sequence = sequence[:length].unsqueeze(0)
            scores = torch.tanh(self.sap_linear(sequence))
            scores = torch.matmul(scores, self.attention).squeeze(dim=2)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            mean = torch.sum(sequence * weights, dim=1)
            # The variance is clamped because it is a difference of two sums
            # that rounding can push below zero.
            variance = torch.sum((sequence**2) * weights, dim=1) - mean**2
            deviation = torch.sqrt(variance.clamp(min=1e-5))
            pooled.append(torch.cat((mean, deviation), dim=1).squeeze(0))
        return torch.stack(pooled)
