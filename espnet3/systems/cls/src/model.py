"""Utterance classifier built on a pre-trained speech encoder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from espnet3.systems.cls.src.pooling import AttentiveStatisticsPooling

__all__ = ["ClassificationHead", "WaveformNormalize", "ClsModel"]

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


class ClassificationHead(nn.Module):
    """Fully connected head mapping a pooled vector to class logits.

    Args:
        input_size: Width of the pooled vector.
        hidden_size: Width of each hidden block.
        num_classes: Number of classes.
        num_layers: Hidden block count.
        dropout: Dropout probability, applied to the input and to every block.

    Examples:
        >>> head = ClassificationHead(2048, 1024, 8)
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


class ClsModel(nn.Module):
    """Single-label utterance classifier over a pre-trained speech encoder.

    A frontend turns the waveform into frames, attentive statistics pooling
    reduces the frames to one vector per utterance, and a fully connected head
    produces class logits. With a fine-tuned WavLM Large frontend this is the
    baseline of the Interspeech 2025 Speech Emotion Recognition in
    Naturalistic Conditions challenge. Only the frontend changes between
    encoders; see ``espnet3.systems.cls.src.frontend``.

    Args:
        frontend: Module mapping ``(speech, speech_lengths)`` to
            ``(frames, frame_lengths)`` with an ``output_size()`` method, such
            as :class:`~espnet3.systems.cls.src.frontend.Wav2Vec2Frontend`.
        num_classes: Number of classes.
        head_size: Width of each hidden block in the head.
        head_layers: Hidden block count in the head.
        dropout: Dropout probability inside the head.
        class_weights: Per-class loss weights in ``token_list`` order, or a
            path to a file holding one weight per line. Usually the inverse
            class frequencies that ``prepare_labels`` writes.
        label_smoothing: Label smoothing of the cross-entropy, as
            ``lsm_weight`` of ESPnet2's classification model. 0 turns it off.

    Examples:
        Selected from a recipe's ``conf/training.yaml``:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.cls.src.model.ClsModel
              frontend:
                _target_: espnet3.systems.cls.src.frontend.Wav2Vec2Frontend
                encoder:
                  _target_: transformers.AutoModel.from_pretrained
                  pretrained_model_name_or_path: microsoft/wavlm-large
              num_classes: 7
              head_size: 1024
    """

    def __init__(
        self,
        frontend: nn.Module,
        num_classes: int,
        head_size: int = 1024,
        head_layers: int = 1,
        dropout: float = 0.5,
        class_weights: Optional[Union[str, Path, Sequence[float]]] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        """Assemble the frontend, the pooling layer and the head."""
        super().__init__()
        self.frontend = frontend
        self.num_classes = int(num_classes)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        # As `lsm_weight` of ESPnet2's classification model
        # (`nn.CrossEntropyLoss(label_smoothing=lsm_weight)`).
        self.label_smoothing = float(label_smoothing)

        self.pooling = AttentiveStatisticsPooling(int(frontend.output_size()))
        self.head = ClassificationHead(
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

    def compute_logits(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Map waveforms to class logits.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.

        Returns:
            torch.Tensor: Logits of shape ``(B, num_classes)``.
        """
        # ESPnet2's CommonPreprocessor returns the speech it reverberates or
        # adds noise to as float64 of shape (samples, 1): back to (B, samples)
        # in the model's dtype.
        if speech.dim() == 3 and speech.size(-1) == 1:
            speech = speech.squeeze(-1)
        if speech.dtype == torch.float64:
            speech = speech.float()
        frames, frame_lengths = self.frontend(speech, speech_lengths)
        return self.head(self.pooling(frames, frame_lengths))

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
            ValueError: If an utterance carries anything other than one label,
                or a label is not one of the model's classes.
        """
        if label.dim() != 2:
            raise ValueError(f"label must be (B, 1), got shape {tuple(label.shape)}")
        if not torch.all(label_lengths == 1):
            raise ValueError(
                "ClsModel is single-label, but the batch carries "
                f"{label_lengths.tolist()} labels per utterance"
            )

        target = label[:, 0].long()
        # A label outside the classes (such as the `<unk>` entry that ends the
        # token list, given to a label the training data never had) would
        # otherwise fail inside cross_entropy as an opaque CUDA assert.
        if bool(((target < 0) | (target >= self.num_classes)).any()):
            raise ValueError(
                f"label holds class indices {sorted(set(target.tolist()))}, but the "
                f"model has {self.num_classes} classes (0..{self.num_classes - 1}). "
                "A label missing from the token list maps to <unk>, which is not a "
                "class; check num_classes against the token list."
            )

        logits = self.compute_logits(speech, speech_lengths)
        loss = F.cross_entropy(
            logits, target, weight=self.class_weights, label_smoothing=self.label_smoothing
        )
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

        The stage reads these to write the waveform statistics that
        :class:`WaveformNormalize` applies, and the per-utterance lengths the
        dataloader batches by. A frontend with a ``logmel`` method (the
        log-mel frontends) also gives its log-mel features as ``logmel``,
        whose statistics (``logmel_stats.npz``) feed ESPnet2's ``GlobalMVN``,
        as ``feats_stats.npz`` does in an ESPnet2 ASR recipe.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.
            **kwargs: Unused extra batch entries.

        Returns:
            Dict[str, torch.Tensor]: The waveforms and their lengths, plus
            the log-mel features and their lengths for a log-mel frontend.
        """
        feats = {"feats": speech, "feats_lengths": speech_lengths}
        logmel = getattr(self.frontend, "logmel", None)
        if callable(logmel):
            feats["logmel"], feats["logmel_lengths"] = logmel(speech, speech_lengths)
        return feats
