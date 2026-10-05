"""Inference model for the eegcls system."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn

from espnet3.systems.eegcls.metrics.scoring_utils import load_class_labels
from espnet3.systems.eegcls.model import EegClsModel

__all__ = ["EegClsInferenceModel"]


class EegClsInferenceModel:
    """Load trained eegcls weights and label one EEG epoch at a time.

    The network is built by Hydra from the inference config, so it must
    match the ``model.network`` block of the training config.

    Args:
        network: Network with the same architecture as in training.
        model_file: Weights saved by the checkpoint-averaging callback.
        token_list: Token list whose final entry is ``<unk>``.
        device: Torch device string, injected by the inference provider.

    Examples:
        Selected from a recipe's ``conf/inference.yaml``:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.eegcls.inference_model.EegClsInferenceModel
              network:
                _target_: braindecode.models.Deep4Net
                n_chans: 62
                n_outputs: 2
                n_times: 1000
                final_conv_length: auto
              model_file: ${exp_dir}/valid.loss.ave_1best.pth
              token_list: ${data_dir}/token_list
    """

    def __init__(
        self,
        network: nn.Module,
        model_file: str | Path,
        token_list: str | Path,
        device: str = "cpu",
    ) -> None:
        """Wrap the network and load the trained weights."""
        self.device = torch.device(device)
        self.classes = load_class_labels(token_list)

        model = EegClsModel(network)
        state = torch.load(str(model_file), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, eeg: Any) -> Tuple[List[int], torch.Tensor, str]:
        """Classify one epoch.

        Args:
            eeg: Epoch as an array or tensor of shape ``(time, channels)``.

        Returns:
            tuple: ``(prediction, scores, label)`` where ``prediction`` holds
            the class index, ``scores`` are per-class probabilities and
            ``label`` is the class name.
        """
        epoch = torch.as_tensor(np.asarray(eeg), dtype=torch.float32)
        epoch = epoch.to(self.device).unsqueeze(0)

        logits = self.model.compute_logits(epoch)
        scores = torch.softmax(logits, dim=-1)[0].cpu()

        prediction = int(scores.argmax())
        return [prediction], scores, self.classes[prediction]
