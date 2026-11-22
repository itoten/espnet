"""Inference model for the cls system."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from espnet2.torch_utils.safe_torch_load import safe_torch_load
from espnet3.systems.esp2_cls.src.scoring_utils import load_class_labels

__all__ = ["ClsInferenceModel"]

Output = Tuple[List[int], torch.Tensor, str]

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class ClsInferenceModel:
    """Load trained cls weights and label one waveform or a batch of them.

    The network is built by Hydra from the inference config, so it must match
    the ``model`` block of the training config.

    Called with one waveform it returns one output; called with a list (the
    inference stage passes a list when the inference config sets
    ``batch_size``) it returns one output per waveform, in the same order. A
    list is sorted by length and cut into sub-batches of at most
    ``max_batch_samples`` samples, padding included, so that waveforms of
    similar length are padded to each other and the GPU gets large batches.

    Args:
        network: ``ClsModel`` with the same architecture as in training.
        model_file: A plain state dict (``*.ave_*best.pth``) or a Lightning
            checkpoint holding one under ``state_dict`` (keys may carry the
            Lightning module's ``model.`` prefix).
        token_list: Token list whose final entry is ``<unk>``.
        device: Torch device string, injected by the inference provider.
        max_batch_samples: Upper bound on ``batch size x longest waveform``
            of one forward pass when called with a list. A single waveform
            longer than this still goes through alone.
        autocast: ``"bf16"`` or ``"fp16"`` to run the network under CUDA
            autocast, as in ``precision: bf16-mixed`` training; ``None``
            computes in fp32. Scores may differ from fp32 in the last digits.

    Examples:
        Selected from a recipe's ``conf/inference.yaml``:

        .. code-block:: yaml

            batch_size: 256
            model:
              _target_: espnet3.systems.cls.src.inference_model.ClsInferenceModel
              network:
                _target_: espnet3.systems.cls.src.model.ClsModel
                frontend:
                  _target_: espnet3.systems.cls.src.frontend.Wav2Vec2Frontend
                  encoder:
                    _target_: transformers.AutoModel.from_pretrained
                    pretrained_model_name_or_path: microsoft/wavlm-large
                num_classes: 7
              model_file: ${exp_dir}/valid.acc.ave_1best.pth
              token_list: ${data_dir}/token_list
              max_batch_samples: 11520000
              autocast: bf16
    """

    def __init__(
        self,
        network: nn.Module,
        model_file: str | Path,
        token_list: str | Path,
        device: str = "cpu",
        max_batch_samples: int = 11_520_000,
        autocast: Optional[str] = None,
    ) -> None:
        """Load the trained weights into the network."""
        self.device = torch.device(device)
        self.classes = load_class_labels(token_list)
        self.max_batch_samples = int(max_batch_samples)
        if autocast is not None and autocast not in _DTYPES:
            raise ValueError(f"autocast must be one of {list(_DTYPES)} or None")
        self.autocast = autocast

        # A full Lightning checkpoint also holds the optimizer and loop state;
        # safe_torch_load explains how to opt in if it cannot be read safely.
        state = safe_torch_load(str(model_file), map_location="cpu")
        state = state.get("state_dict", state)
        # Weights saved from the Lightning module carry its `model.` prefix.
        if state and all(key.startswith("model.") for key in state):
            state = {key[len("model.") :]: value for key, value in state.items()}
        # The loss weights are a training-only buffer; the inference network
        # is built without them.
        state.pop("class_weights", None)
        network.load_state_dict(state)
        self.model = network.to(self.device).eval()

    def _autocast(self):
        if self.autocast is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=_DTYPES[self.autocast])

    def _sub_batches(self, lengths: Sequence[int]) -> List[List[int]]:
        """Group indices, longest first, under ``max_batch_samples``."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        groups: List[List[int]] = []
        for i in order:
            # The first index of a group is its longest waveform.
            if (
                groups
                and (len(groups[-1]) + 1) * lengths[groups[-1][0]]
                <= self.max_batch_samples
            ):
                groups[-1].append(i)
            else:
                groups.append([i])
        return groups

    @torch.no_grad()
    def _scores(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """Return the class probabilities of equally treated waveforms."""
        lengths = torch.tensor([w.numel() for w in waveforms])
        batch = torch.zeros(len(waveforms), int(lengths.max()))
        for row, wave in enumerate(waveforms):
            batch[row, : wave.numel()] = wave
        batch = batch.to(self.device, non_blocking=True)
        with self._autocast():
            logits = self.model.compute_logits(batch, lengths.to(self.device))
        return torch.softmax(logits.float(), dim=-1).cpu()

    def _output(self, scores: torch.Tensor) -> Output:
        prediction = int(scores.argmax())
        return [prediction], scores, self.classes[prediction]

    def __call__(self, speech: Any) -> Output | List[Output]:
        """Classify one waveform or a list of them.

        Args:
            speech: Waveform as an array or tensor of shape ``(samples,)``,
                or a list of them.

        Returns:
            ``(prediction, scores, label)`` where ``prediction`` holds the
            class index, ``scores`` are per-class probabilities and ``label``
            is the class name; a list of them, in input order, for a list.
        """
        if isinstance(speech, (list, tuple)):
            waveforms = [
                torch.as_tensor(np.asarray(s), dtype=torch.float32) for s in speech
            ]
            outputs: List[Optional[Output]] = [None] * len(waveforms)
            for group in self._sub_batches([w.numel() for w in waveforms]):
                scores = self._scores([waveforms[i] for i in group])
                for row, i in enumerate(group):
                    outputs[i] = self._output(scores[row])
            return outputs
        waveform = torch.as_tensor(np.asarray(speech), dtype=torch.float32)
        return self._output(self._scores([waveform])[0])
