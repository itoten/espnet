"""Inference entry point for the wavlmasp recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.systems.wavlmasp.metrics.scoring_utils import load_class_labels

__all__ = ["EmotionClassification", "build_output"]


class EmotionClassification:
    """Load a trained wavlmasp model and label one waveform at a time.

    Mirrors the return shape of ``espnet2.bin.cls_inference.Classification`` so
    the recipe's ``output_fn`` stays the same.

    Args:
        train_config: Training config written next to the checkpoint.
        model_file: Weights saved by the checkpoint-averaging callback.
        token_list: Token list whose final entry is ``<unk>``.
        device: Torch device string.

    Examples:
        Selected from a recipe's ``conf/inference.yaml``:

        .. code-block:: yaml

            model:
              _target_: src.inference.EmotionClassification
              train_config: ${exp_dir}/config.yaml
              model_file: ${exp_dir}/valid.acc.ave_1best.pth
              token_list: ${data_dir}/token_list
    """

    def __init__(
        self,
        train_config: str | Path,
        model_file: str | Path,
        token_list: str | Path,
        device: str = "cpu",
    ) -> None:
        """Rebuild the model from its training config and load the weights."""
        self.device = torch.device(device)
        self.classes = load_class_labels(token_list)

        config = OmegaConf.load(str(train_config))
        model = instantiate(config.model)
        state = torch.load(str(model_file), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

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

        frames, frame_lengths = self.model._encode(waveform, lengths)
        logits = self.model.head(self.model.pooling(frames, frame_lengths))
        scores = torch.softmax(logits, dim=-1)[0].cpu()

        prediction = int(scores.argmax())
        return [prediction], scores, self.classes[prediction]


def build_output(data: dict, model_output: Sequence, idx: int) -> dict:
    """Build one SCP record from a model output."""
    _, scores, hyp = model_output
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": hyp,
        "ref": data.get("label", ""),
        "score": " ".join(str(score) for score in scores.tolist()),
    }
