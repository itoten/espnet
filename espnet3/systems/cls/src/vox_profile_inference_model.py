"""Zero-shot inference with the Vox-Profile Whisper emotion classifier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from espnet3.systems.cls.metrics.scoring_utils import load_class_labels
from espnet3.systems.cls.src.label_map import (
    gather_class_scores,
    resolve_label_map,
)

__all__ = ["VoxProfileEmotionInferenceModel"]

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_SECONDS = 15
ENCODER_PREFIX = "backbone_model.encoder."


class _EmotionHead(nn.Module):
    """Point-wise convolutions, then the categorical-emotion classifier.

    The submodule names and indices follow the released checkpoint
    (``model_seq.{0,3,6}``, ``emotion_layer.{0,2}``) so that its tensors load
    by name. Dropout layers keep their indices and do nothing in eval mode.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int) -> None:
        super().__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        self.emotion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, frames: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
        """Return logits from encoder frames ``(batch, time, dim)``.

        Each utterance is averaged over its own ``lengths[i]`` frames, so the
        padding up to 15 s is left out of the mean.
        """
        hidden = self.model_seq(frames.transpose(1, 2)).transpose(1, 2)
        pooled = torch.stack(
            [hidden[i, :n].mean(dim=0) for i, n in enumerate(lengths)]
        )
        return self.emotion_layer(pooled)


class VoxProfileEmotionInferenceModel:
    """Label waveforms with the Vox-Profile emotion classifier, as released.

    Vox-Profile (Feng et al., arXiv 2505.14648) releases a categorical
    emotion classifier trained on MSP-Podcast only: the Whisper Large-v3
    encoder, adapted with LoRA on the feed-forward layers of its upper half,
    followed by point-wise convolutions, mean pooling over time and a
    two-layer classifier over nine labels (Anger, Contempt, Disgust, Fear,
    Happiness, Neutral, Sadness, Surprise, Other). This class rebuilds that
    network from ``transformers``' ``WhisperEncoder`` and the released
    weights, without the authors' package, and maps its labels onto the
    recipe's classes so that the ``infer`` and ``measure`` stages score it
    as they score a trained ``ClsModel``. Use it with a standalone inference
    config, which carries its own ``exp_tag``.

    The forward pass follows the released code:

    * The waveform is turned into 128 log-mel features by Whisper's feature
      extractor with 15-second chunks: longer audio is cut at 15 s and
      shorter audio is zero-padded to 15 s (1,500 feature frames, 750
      encoder frames). The encoder's position table is cut to 750 rows.
    * LoRA is merged into the weights: ``W + B @ A * alpha / r`` with
      ``loralib``'s default ``alpha = 1`` and the checkpoint's rank, which is
      what ``loralib.Linear`` computes in eval mode.
    * The classifier reads the encoder's last hidden state (after its final
      layer norm), and the mean over time covers the utterance's own frames,
      ``((samples // 160) - 1) // 2 + 1``, at most 750. An utterance shorter
      than 160 samples counts as one frame here, where the released code
      would average over none.

    Only the categorical-emotion head is loaded; the checkpoint's other
    heads (detailed emotion, arousal, valence, dominance) and the Whisper
    decoder it carries are not read.

    Scores are restricted to the recipe's classes as for the other zero-shot
    models: the probabilities of the labels named in ``label_map`` are
    gathered in ``token_list`` order and renormalised, and the rest
    (``Contempt``, ``Other``) are discarded.

    The released weights are under the OpenRAIL license and are not for
    commercial use; see the model card.

    Args:
        token_list: Token list whose final entry is ``<unk>``; it fixes the
            classes and their order in the score vector.
        label_map: Mapping from a Vox-Profile label (e.g. ``Anger``) to a
            recipe class.
        model_tag: Hugging Face repository of the released classifier.
        revision: Repository revision (branch, tag or commit); ``None`` for
            the default branch.
        backbone: Hugging Face repository whose config and feature extractor
            describe the encoder (its weights are not loaded).
        dtype: Torch dtype name for the encoder and head; the weights are
            released in ``float32``.
        attn_implementation: Attention implementation of the encoder
            (``sdpa`` or ``eager``).
        device: Torch device string, injected by the inference provider.

    Raises:
        ValueError: If the checkpoint is not a Whisper Large categorical
            classifier, or ``label_map`` names an unknown label or class.

    Examples:
        Selected from a recipe's standalone inference config:

        .. code-block:: yaml

            exp_tag: zeroshot_vox_profile_whisper
            model:
              _target_: espnet3.systems.cls.src.vox_profile_inference_model.VoxProfileEmotionInferenceModel
              token_list: ${data_dir}/token_list
              label_map:
                Anger: angry
                Happiness: happy
    """

    LABELS = (
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
        "Other",
    )

    def __init__(
        self,
        token_list: str | Path,
        label_map: Mapping[str, str],
        model_tag: str = "tiantiaf/whisper-large-v3-msp-podcast-emotion",
        revision: Optional[str] = None,
        backbone: str = "openai/whisper-large-v3",
        dtype: str = "float32",
        attn_implementation: str = "sdpa",
        device: str = "cpu",
    ) -> None:
        """Rebuild the network, load the released weights and map labels."""
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.classes = load_class_labels(token_list)
        self.source_index = resolve_label_map(
            self.LABELS, dict(label_map), self.classes, "Vox-Profile"
        )

        with open(
            hf_hub_download(model_tag, "config.json", revision=revision)
        ) as f:
            model_config = json.load(f)
        self._check_model_config(model_config)

        config = WhisperConfig.from_pretrained(backbone)
        config.max_source_positions = MAX_SECONDS * SAMPLE_RATE // HOP_LENGTH // 2
        config._attn_implementation = attn_implementation
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            backbone, chunk_length=MAX_SECONDS
        )

        weights = hf_hub_download(model_tag, "model.safetensors", revision=revision)
        encoder_state, head_state = self._read_weights(
            weights, int(model_config["lora_rank"]), model_config["finetune_method"]
        )
        encoder = WhisperEncoder(config)
        encoder.load_state_dict(encoder_state, strict=True)
        head = _EmotionHead(
            config.d_model,
            int(model_config["hidden_dim"]),
            int(model_config["output_class_num"]),
        )
        head.load_state_dict(head_state, strict=True)
        self.encoder = encoder.to(self.device, self.dtype).eval()
        self.head = head.to(self.device, self.dtype).eval()

    def _check_model_config(self, model_config: Dict[str, Any]) -> None:
        """Fail early on a checkpoint this class does not rebuild."""
        expected = {
            "pretrain_model": "whisper_large",
            "output_class_num": len(self.LABELS),
            "use_conv_output": True,
        }
        wrong = {
            key: model_config.get(key)
            for key, value in expected.items()
            if model_config.get(key) != value
        }
        if wrong or model_config.get("finetune_method") not in ("lora", "finetune"):
            raise ValueError(
                "Expected a Vox-Profile Whisper Large categorical emotion "
                f"checkpoint ({expected}, finetune_method lora or finetune); "
                f"got {model_config}"
            )

    @staticmethod
    def _read_weights(
        path: str, lora_rank: int, finetune_method: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Read the encoder and emotion-head tensors, merging LoRA.

        Args:
            path: The released ``model.safetensors``.
            lora_rank: LoRA rank from the checkpoint's config.
            finetune_method: ``lora`` merges ``lora_A`` and ``lora_B``.

        Returns:
            tuple: ``(encoder_state, head_state)`` keyed for
            ``WhisperEncoder`` and ``_EmotionHead``.
        """
        encoder_state: Dict[str, torch.Tensor] = {}
        head_state: Dict[str, torch.Tensor] = {}
        lora: Dict[str, Dict[str, torch.Tensor]] = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(ENCODER_PREFIX):
                    name = key[len(ENCODER_PREFIX) :]
                    if name.endswith((".lora_A", ".lora_B")):
                        module, part = name.rsplit(".", 1)
                        lora.setdefault(module, {})[part] = f.get_tensor(key)
                    else:
                        encoder_state[name] = f.get_tensor(key)
                elif key.startswith(("model_seq.", "emotion_layer.")):
                    head_state[key] = f.get_tensor(key)

        if lora and finetune_method != "lora":
            raise ValueError("The checkpoint has LoRA tensors but is not a LoRA model.")
        scaling = 1.0 / lora_rank if lora else 0.0  # loralib: alpha (1) / r
        for module, parts in lora.items():
            weight = encoder_state[f"{module}.weight"]
            encoder_state[f"{module}.weight"] = (
                weight + (parts["lora_B"] @ parts["lora_A"]) * scaling
            )
        if lora:
            logger.info("Merged LoRA (rank %d) into %d layers.", lora_rank, len(lora))
        return encoder_state, head_state

    @staticmethod
    def frame_lengths(num_samples: Sequence[int]) -> List[int]:
        """Encoder frames covered by each waveform, as in the released code.

        Args:
            num_samples: Waveform lengths in samples.

        Returns:
            list[int]: ``((n // 160) - 1) // 2 + 1``, clipped to ``[1, 750]``.

        Examples:
            >>> VoxProfileEmotionInferenceModel.frame_lengths([16000, 480000])
            [50, 750]
        """
        limit = MAX_SECONDS * SAMPLE_RATE // HOP_LENGTH // 2
        return [
            min(max(((n // HOP_LENGTH) - 1) // 2 + 1, 1), limit) for n in num_samples
        ]

    @torch.no_grad()
    def label_probabilities(self, speech: Sequence[Any]) -> torch.Tensor:
        """Return the model's probabilities over its nine labels.

        Args:
            speech: 16 kHz mono waveforms.

        Returns:
            torch.Tensor: ``(batch, 9)`` softmax probabilities in
            :attr:`LABELS` order, on the CPU in ``float32``.
        """
        waveforms = [_to_numpy(waveform) for waveform in speech]
        features = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            max_length=MAX_SECONDS * SAMPLE_RATE,
        ).input_features
        frames = self.encoder(
            features.to(self.device, self.dtype)
        ).last_hidden_state
        logits = self.head(frames, self.frame_lengths([len(w) for w in waveforms]))
        return torch.softmax(logits.float(), dim=-1).cpu()

    def __call__(self, speech: Any) -> Any:
        """Classify one waveform, or a batch of them.

        Args:
            speech: One 16 kHz mono waveform (array or tensor of shape
                ``(samples,)``), or a list of them.

        Returns:
            ``(prediction, scores, label)`` for one waveform, the same as
            ``ClsInferenceModel``: ``prediction`` holds the class index,
            ``scores`` the per-class probabilities in ``token_list`` order
            and ``label`` the class name. A list of such tuples for a batch.
        """
        batched = isinstance(speech, (list, tuple))
        probabilities = self.label_probabilities(speech if batched else [speech])
        outputs = []
        for row in probabilities:
            scores = gather_class_scores(row, self.source_index)
            prediction = int(scores.argmax())
            outputs.append(([prediction], scores, self.classes[prediction]))
        return outputs if batched else outputs[0]


def _to_numpy(waveform: Any) -> np.ndarray:
    """Return a waveform as a 1-D float32 array."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    return np.asarray(waveform, dtype=np.float32).reshape(-1)
