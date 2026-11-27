"""Zero-shot inference with the Vox-Profile WavLM emotion classifier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

from espnet3.systems.cls.src.label_map import resolve_label_map
from espnet3.systems.cls.src.vox_profile_inference_model import (
    MAX_SECONDS,
    SAMPLE_RATE,
    VoxProfileEmotionInferenceModel,
    _EmotionHead,
    _to_numpy,
)
from espnet3.systems.esp2_cls.src.scoring_utils import load_class_labels

__all__ = ["VoxProfileWavLMEmotionInferenceModel"]

logger = logging.getLogger(__name__)

BACKBONE_PREFIX = "backbone_model."


class VoxProfileWavLMEmotionInferenceModel(VoxProfileEmotionInferenceModel):
    """Label waveforms with the Vox-Profile WavLM emotion classifier, as released.

    The WavLM Large counterpart of :class:`VoxProfileEmotionInferenceModel`
    (Feng et al., arXiv 2505.14648): trained on MSP-Podcast only, the same
    nine labels, and the same point-wise convolution head. The released
    network (``WavLMWrapper`` of ``vox-profile-release``) is rebuilt from
    ``transformers``' ``WavLMModel`` and the released weights, without the
    authors' package. The ``infer`` and ``measure`` stages score it as they
    score a trained ``ClsModel``; the scores are restricted to the recipe's
    classes in the same way as for the Whisper model.

    The forward pass follows the released code:

    * Each waveform is cut at 15 s, normalised to zero mean and unit variance
      by ``Wav2Vec2FeatureExtractor`` (as ``microsoft/wavlm-large`` does) and
      run through the encoder on its own, so there is no padding.
    * The classifier reads the softmax-weighted sum of all 25 hidden states
      (the convolutional output and the 24 layers; the last one is the
      encoder's output after its final layer norm) with the checkpoint's
      ``weights``, then the head, then the mean over time.
    * The released checkpoint is a full fine-tune without LoRA
      (``finetune_method: finetune``); the encoder is not frozen to the
      ``microsoft/wavlm-large`` weights but read from the checkpoint.

    The released weights are under the OpenRAIL license and are not for
    commercial use; see the model card.

    Args:
        token_list: Token list whose final entry is ``<unk>``.
        label_map: Mapping from a Vox-Profile label to a recipe class.
        model_tag: Hugging Face repository of the released classifier.
        revision: Repository revision (branch, tag or commit).
        backbone: Hugging Face repository whose config and feature extractor
            describe the encoder (its weights are not loaded).
        dtype: Torch dtype name for the encoder and head.
        device: Torch device string, injected by the inference provider.

    Raises:
        ValueError: If the checkpoint is not a WavLM Large categorical
            classifier, or ``label_map`` names an unknown label or class.
    """

    def __init__(
        self,
        token_list: str | Path,
        label_map: Mapping[str, str],
        model_tag: str = "tiantiaf/wavlm-large-categorical-emotion",
        revision: Optional[str] = None,
        backbone: str = "microsoft/wavlm-large",
        dtype: str = "float32",
        device: str = "cpu",
    ) -> None:
        """Rebuild the network, load the released weights and map labels."""
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.classes = load_class_labels(token_list)
        self.source_index = resolve_label_map(
            self.LABELS, dict(label_map), self.classes, "Vox-Profile"
        )

        with open(hf_hub_download(model_tag, "config.json", revision=revision)) as f:
            model_config = json.load(f)
        self._check_model_config(model_config)

        config = WavLMConfig.from_pretrained(backbone)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(backbone)
        weights = hf_hub_download(model_tag, "model.safetensors", revision=revision)
        encoder_state, layer_weights, head_state = self._read_weights(weights)

        encoder = WavLMModel(config)
        encoder.load_state_dict(encoder_state, strict=True)
        head = _EmotionHead(
            config.hidden_size,
            int(model_config["hidden_dim"]),
            int(model_config["output_class_num"]),
        )
        head.load_state_dict(head_state, strict=True)
        if layer_weights.numel() != config.num_hidden_layers + 1:
            raise ValueError(
                f"Expected {config.num_hidden_layers + 1} layer weights, "
                f"got {layer_weights.numel()}."
            )
        self.encoder = encoder.to(self.device, self.dtype).eval()
        self.head = head.to(self.device, self.dtype).eval()
        self.layer_weights = layer_weights.to(self.device, self.dtype)

    def _check_model_config(self, model_config: Dict[str, Any]) -> None:
        """Fail early on a checkpoint this class does not rebuild."""
        expected = {
            "pretrain_model": "wavlm_large",
            "output_class_num": len(self.LABELS),
            "use_conv_output": True,
            "finetune_method": "finetune",
        }
        wrong = {
            key: model_config.get(key)
            for key, value in expected.items()
            if model_config.get(key) != value
        }
        if wrong:
            raise ValueError(
                "Expected a Vox-Profile WavLM Large categorical emotion "
                f"checkpoint ({expected}); got {model_config}"
            )

    @staticmethod
    def _read_weights(
        path: str,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Read the encoder, the layer weights and the emotion head.

        Returns:
            tuple: ``(encoder_state, layer_weights, head_state)``. The other
            heads (detailed emotion, arousal, valence, dominance) are not read.
        """
        encoder_state: Dict[str, torch.Tensor] = {}
        head_state: Dict[str, torch.Tensor] = {}
        layer_weights = None
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(BACKBONE_PREFIX):
                    encoder_state[key[len(BACKBONE_PREFIX) :]] = f.get_tensor(key)
                elif key == "weights":
                    layer_weights = f.get_tensor(key)
                elif key.startswith(("model_seq.", "emotion_layer.")):
                    head_state[key] = f.get_tensor(key)
        if layer_weights is None:
            raise ValueError("The checkpoint has no layer weights (`weights`).")
        return encoder_state, layer_weights, head_state

    @torch.no_grad()
    def label_probabilities(self, speech: Sequence[Any]) -> torch.Tensor:
        """Return the model's probabilities over its nine labels.

        Args:
            speech: 16 kHz mono waveforms.

        Returns:
            torch.Tensor: ``(batch, 9)`` softmax probabilities in
            :attr:`LABELS` order, on the CPU in ``float32``.
        """
        rows = []
        for waveform in speech:
            samples = _to_numpy(waveform)[: MAX_SECONDS * SAMPLE_RATE]
            values = self.feature_extractor(
                samples, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_values
            out = self.encoder(
                values.to(self.device, self.dtype), output_hidden_states=True
            )
            # The released code ran on transformers 4.46, whose last hidden state is
            # the encoder's output (after its final layer norm, for WavLM Large).
            # Newer transformers return the last layer's output before that norm in
            # ``hidden_states``, so the last entry is replaced by ``last_hidden_state``.
            hidden = list(out.hidden_states[:-1]) + [out.last_hidden_state]
            stacked = torch.stack(hidden, dim=0)  # (25, 1, time, dim)
            weights = torch.softmax(self.layer_weights, dim=-1)
            frames = (weights[:, None, None, None] * stacked).sum(dim=0)
            logits = self.head(frames, [frames.shape[1]])
            rows.append(torch.softmax(logits.float(), dim=-1).cpu())
        return torch.cat(rows, dim=0)
