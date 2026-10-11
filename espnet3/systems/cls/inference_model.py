"""Inference model for the cls system."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn

from espnet3.systems.cls.metrics.scoring_utils import load_class_labels

__all__ = ["ClsInferenceModel"]


class ClsInferenceModel:
    """Load trained cls weights and label one waveform at a time.

    The network is built by Hydra from the inference config, so it must match
    the ``model`` block of the training config.

    Args:
        network: ``ClsModel`` with the same architecture as in training.
        model_file: A plain state dict (``*.ave_*best.pth``) or a Lightning
            checkpoint holding one under ``state_dict``.
        token_list: Token list whose final entry is ``<unk>``.
        device: Torch device string, injected by the inference provider.

    Examples:
        Selected from a recipe's ``conf/inference.yaml``:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.cls.inference_model.ClsInferenceModel
              network:
                _target_: espnet3.systems.cls.model.ClsModel
                frontend:
                  _target_: espnet3.systems.cls.frontend.Wav2Vec2Frontend
                  encoder:
                    _target_: transformers.AutoModel.from_pretrained
                    pretrained_model_name_or_path: microsoft/wavlm-large
                num_classes: 7
              model_file: ${exp_dir}/valid.acc.ave_1best.pth
              token_list: ${data_dir}/token_list
    """

    def __init__(
        self,
        network: nn.Module,
        model_file: str | Path,
        token_list: str | Path,
        device: str = "cpu",
    ) -> None:
        """Load the trained weights into the network."""
        self.device = torch.device(device)
        self.classes = load_class_labels(token_list)

        state = torch.load(str(model_file), map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        # The loss weights are a training-only buffer; the inference network
        # is built without them.
        state.pop("class_weights", None)
        network.load_state_dict(state)
        self.model = network.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, speech: Any) -> Tuple[List[int], torch.Tensor, str]:
        """Classify one waveform.

        Args:
            speech: Waveform as an array or tensor of shape ``(samples,)``.

        Returns:
            tuple: ``(prediction, scores, label)`` where ``prediction`` holds
            the class index, ``scores`` are per-class probabilities and
            ``label`` is the class name.
        """
        waveform = torch.as_tensor(np.asarray(speech), dtype=torch.float32)
        waveform = waveform.to(self.device).unsqueeze(0)
        lengths = torch.tensor([waveform.size(1)], device=self.device)

        logits = self.model.compute_logits(waveform, lengths)
        scores = torch.softmax(logits, dim=-1)[0].cpu()

        prediction = int(scores.argmax())
        return [prediction], scores, self.classes[prediction]
