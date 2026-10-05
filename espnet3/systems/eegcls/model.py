"""EEG epoch classifier that wraps any ``(batch, channels, time)`` network."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["EegClsModel"]


class EegClsModel(nn.Module):
    """Single-label classifier over an EEG network.

    The network is built from the config, so this class holds no channel
    count, class count or sampling rate. It only fits the network to what
    ESPnet3's trainer expects: ``forward`` returns ``(loss, stats, weight)``
    and ``collect_feats`` exposes the input for ``collect_stats``.

    Args:
        network: Module mapping ``(B, channels, time)`` to class logits of
            shape ``(B, classes)``. The logits must not be passed through a
            softmax.

    Examples:
        Selected from a recipe's ``conf/training.yaml``:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.eegcls.model.EegClsModel
              network:
                _target_: braindecode.models.Deep4Net
                n_chans: 62
                n_outputs: 2
                n_times: 1000
    """

    def __init__(self, network: nn.Module) -> None:
        """Keep the network."""
        super().__init__()
        self.network = network

    def compute_logits(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of epochs.

        Args:
            eeg: Epochs of shape ``(B, time, channels)``.

        Returns:
            torch.Tensor: Logits of shape ``(B, classes)``.
        """
        # (time, channels) because CommonCollateFn pads the first dimension.
        return self.network(eeg.transpose(1, 2))

    def forward(
        self,
        eeg: torch.Tensor,
        eeg_lengths: torch.Tensor,
        label: torch.Tensor,
        label_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Classify a batch and return its loss.

        Args:
            eeg: Epochs of shape ``(B, time, channels)``.
            eeg_lengths: Valid time steps per epoch, shape ``(B,)``. Unused,
                since every epoch of a recipe has the same length.
            label: Class indices of shape ``(B, 1)``.
            label_lengths: Label count per epoch, shape ``(B,)``. Every entry
                must be 1, since the model is single-label.
            **kwargs: Unused extra batch entries.

        Returns:
            tuple: ``(loss, stats, weight)`` as ESPnet3's trainer expects.

        Raises:
            ValueError: If an epoch carries anything other than one label.
        """
        if label.dim() != 2:
            raise ValueError(f"label must be (B, 1), got shape {tuple(label.shape)}")
        if not torch.all(label_lengths == 1):
            raise ValueError(
                "EegClsModel is single-label, but the batch carries "
                f"{label_lengths.tolist()} labels per epoch"
            )

        logits = self.compute_logits(eeg)
        target = label[:, 0].long()

        loss = F.cross_entropy(logits, target)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        stats = {"loss": loss.detach(), "acc": accuracy.detach()}
        weight = torch.tensor(eeg.size(0), dtype=loss.dtype, device=loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        eeg: torch.Tensor,
        eeg_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Expose the epochs so ``collect_stats`` can record their shape.

        Args:
            eeg: Epochs of shape ``(B, time, channels)``.
            eeg_lengths: Valid time steps per epoch, shape ``(B,)``.
            **kwargs: Unused extra batch entries.

        Returns:
            Dict[str, torch.Tensor]: The epochs and their lengths.
        """
        return {"feats": eeg, "feats_lengths": eeg_lengths}
