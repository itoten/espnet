"""Speech emotion classifier built on a self-supervised speech encoder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from espnet3.systems.wavlmasp.pooling import AttentiveStatisticsPooling

__all__ = ["EmotionHead", "WaveformNormalize", "WavlmAspModel"]

logger = logging.getLogger(__name__)


class WaveformNormalize(nn.Module):
    """Standardise a waveform with statistics taken over the training set.

    ``espnet2.layers.global_mvn.GlobalMVN`` cannot be used here: it masks
    padding through ``make_pad_mask``, which builds a square matrix over the
    sequence. That is affordable for a few hundred frames and not for the
    hundred thousand samples of a waveform.

    Args:
        stats_file: ``feats_stats.npz`` written by ``collect_stats``, holding
            ``count``, ``sum`` and ``sum_square`` over the raw waveform.
        eps: Floor on the deviation.

    Examples:
        >>> normalize = WaveformNormalize("exp/stats/train/feats_stats.npz")
        >>> lengths = torch.tensor([16000, 8000])
        >>> speech, lengths = normalize(torch.randn(2, 16000), lengths)
    """

    def __init__(self, stats_file: Union[str, Path], eps: float = 1.0e-6) -> None:
        """Read the statistics and keep the mean and deviation as buffers."""
        stats = np.load(str(stats_file))
        count = float(stats["count"])
        mean = float(stats["sum"]) / count
        variance = float(stats["sum_square"]) / count - mean * mean

        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer(
            "std", torch.tensor(max(variance, 0.0) ** 0.5 + eps, dtype=torch.float)
        )

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standardise the waveform.

        Padding is left as it is: the encoder is told the true lengths through
        its attention mask, so the padded tail never reaches the pooling.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.

        Returns:
            tuple: The standardised waveforms and their unchanged lengths.
        """
        return (speech - self.mean) / self.std, speech_lengths


class EmotionHead(nn.Module):
    """Fully connected head mapping a pooled vector to class logits.

    Args:
        input_size: Width of the pooled vector.
        hidden_size: Width of each hidden block.
        num_classes: Number of emotion classes.
        num_layers: Hidden block count.
        dropout: Dropout probability, applied to the input and to every block.

    Examples:
        >>> head = EmotionHead(2048, 1024, 8)
        >>> head(torch.randn(2, 2048)).shape
        torch.Size([2, 8])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        """Build the dropout, the hidden blocks and the output projection."""
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.input_dropout = nn.Dropout(dropout)
        blocks = [
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        ]
        for _ in range(num_layers - 1):
            blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Map a pooled vector to logits.

        Args:
            pooled: Pooled representations of shape ``(B, input_size)``.

        Returns:
            torch.Tensor: Logits of shape ``(B, num_classes)``.
        """
        hidden = self.input_dropout(pooled)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class WavlmAspModel(nn.Module):
    """Single-label emotion classifier over a self-supervised encoder.

    The encoder runs on the raw waveform, attentive statistics pooling reduces
    the frame sequence to one vector per utterance, and a fully connected head
    produces class logits. This is the baseline of the Interspeech 2025 Speech
    Emotion Recognition in Naturalistic Conditions challenge.

    The convolutional feature extractor of the encoder is frozen; its
    transformer layers are trained.

    Args:
        encoder: Hugging Face speech encoder, normally instantiated from
            ``transformers.AutoModel``.
        num_classes: Number of emotion classes.
        head_size: Width of each hidden block in the head.
        head_layers: Hidden block count in the head.
        dropout: Dropout probability inside the head.
        class_weights: Per-class loss weights in ``token_list`` order, or a
            path to a file holding one weight per line. A class imbalance is
            the norm for emotion corpora, so these are usually the inverse
            class frequencies that ``prepare_labels`` writes.
        normalize: Optional module applied to the waveform before the encoder,
            normally :class:`WaveformNormalize` reading the statistics that
            ``collect_stats`` wrote.
        freeze_feature_extractor: Whether to freeze the encoder's convolutional
            feature extractor.

    Examples:
        Selected from a recipe's ``conf/training.yaml``:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.wavlmasp.model.WavlmAspModel
              encoder:
                _target_: transformers.AutoModel.from_pretrained
                pretrained_model_name_or_path: microsoft/wavlm-large
              num_classes: 7
              head_size: 1024
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        head_size: int = 1024,
        head_layers: int = 1,
        dropout: float = 0.5,
        class_weights: Optional[Union[str, Path, Sequence[float]]] = None,
        normalize: Optional[nn.Module] = None,
        freeze_feature_extractor: bool = True,
    ) -> None:
        """Assemble the encoder, the pooling layer and the head."""
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        self.num_classes = int(num_classes)

        if freeze_feature_extractor:
            if not hasattr(encoder, "freeze_feature_encoder"):
                raise AttributeError(
                    "freeze_feature_extractor is set but the encoder has no "
                    "freeze_feature_encoder(). Pass freeze_feature_extractor=false "
                    "for an encoder without a convolutional feature extractor."
                )
            encoder.freeze_feature_encoder()

        encoder_size = int(encoder.config.hidden_size)
        self.pooling = AttentiveStatisticsPooling(encoder_size)
        self.head = EmotionHead(
            self.pooling.output_size,
            head_size,
            self.num_classes,
            num_layers=head_layers,
            dropout=dropout,
        )

        # `from_pretrained` hands back a model in eval mode, and Lightning does
        # not switch modes when training starts. Left alone, the encoder would
        # train with its dropout disabled.

        self.train()

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            # A path is read here rather than in the config, because the file
            # comes from prepare_labels, a stage that runs after the config is
            # loaded.
            if isinstance(class_weights, (str, Path)):
                class_weights = Path(class_weights).read_text().split()
            weights = torch.tensor(
                [float(value) for value in class_weights], dtype=torch.float
            )
            if weights.numel() != self.num_classes:
                raise ValueError(
                    f"class_weights holds {weights.numel()} values but the model "
                    f"has {self.num_classes} classes"
                )
            self.register_buffer("class_weights", weights)

    def _encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder and report how many frames each utterance produced."""
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        mask = (
            torch.arange(speech.size(1), device=speech.device)[None, :]
            < speech_lengths[:, None]
        ).long()
        frames = self.encoder(speech, attention_mask=mask).last_hidden_state
        # The encoder owns its convolutional strides, so it is the only thing
        # that can say how many frames a waveform becomes.
        frame_lengths = self.encoder._get_feat_extract_output_lengths(speech_lengths)
        return frames, frame_lengths.to(torch.long)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        label: torch.Tensor,
        label_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Classify a batch and return its loss.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.
            label: Class indices of shape ``(B, 1)``.
            label_lengths: Label count per utterance, shape ``(B,)``. Every
                entry must be 1, since the model is single-label.
            **kwargs: Unused extra batch entries.

        Returns:
            tuple: ``(loss, stats, weight)`` as ESPnet3's trainer expects.

        Raises:
            ValueError: If an utterance carries anything other than one label.
        """
        if label.dim() != 2:
            raise ValueError(f"label must be (B, 1), got shape {tuple(label.shape)}")
        if not torch.all(label_lengths == 1):
            raise ValueError(
                "WavlmAspModel is single-label, but the batch carries "
                f"{label_lengths.tolist()} labels per utterance"
            )

        frames, frame_lengths = self._encode(speech, speech_lengths)
        logits = self.head(self.pooling(frames, frame_lengths))
        target = label[:, 0].long()

        loss = F.cross_entropy(logits, target, weight=self.class_weights)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        stats = {"loss": loss.detach(), "acc": accuracy.detach()}
        weight = torch.tensor(speech.size(0), dtype=loss.dtype, device=loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Expose the waveform so ``collect_stats`` can measure it.

        The stage reads these to write the waveform statistics the ``normalize``
        module applies, and the per-utterance lengths the dataloader batches by.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.
            **kwargs: Unused extra batch entries.

        Returns:
            Dict[str, torch.Tensor]: The waveforms and their lengths.
        """
        return {"feats": speech, "feats_lengths": speech_lengths}
