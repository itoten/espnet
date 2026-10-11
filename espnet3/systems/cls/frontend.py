"""Speech encoders that turn waveforms into frame features for the classifier.

Every frontend takes a batch of waveforms with their lengths and returns frame
features with their lengths, the interface of ESPnet2's ``AbsFrontend``. Each
one feeds its encoder the input that encoder was pre-trained on, so only the
encoder differs between models built on different frontends.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from espnet2.asr.frontend.abs_frontend import AbsFrontend

__all__ = [
    "OwsmFrontend",
    "W2vBertFrontend",
    "Wav2Vec2Frontend",
    "WhisperFrontend",
    "load_s2t_model",
]


def _split_batch(speech: torch.Tensor, speech_lengths: torch.Tensor) -> list:
    """Cut the padding off each waveform and move it to the CPU as numpy."""
    return [
        wave[:length].detach().float().cpu().numpy()
        for wave, length in zip(speech, speech_lengths.tolist())
    ]


class Wav2Vec2Frontend(AbsFrontend):
    """Hugging Face wav2vec 2.0-family encoder on the raw waveform.

    Covers WavLM, HuBERT, wav2vec 2.0 and XLS-R, which share the same front:
    a strided convolutional feature encoder over the waveform followed by
    transformer layers.

    Args:
        encoder: Hugging Face model, normally from
            ``transformers.AutoModel.from_pretrained``.
        normalize: Optional module standardising the waveform before the
            encoder, normally ``WaveformNormalize``.
        freeze_subsampling: Whether to freeze the convolutional feature
            encoder.

    Examples:
        .. code-block:: yaml

            frontend:
              _target_: espnet3.systems.cls.frontend.Wav2Vec2Frontend
              encoder:
                _target_: transformers.AutoModel.from_pretrained
                pretrained_model_name_or_path: microsoft/wavlm-large
    """

    def __init__(
        self,
        encoder: nn.Module,
        normalize: Optional[nn.Module] = None,
        freeze_subsampling: bool = True,
    ) -> None:
        """Keep the encoder and optionally freeze its convolutional front."""
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        if freeze_subsampling:
            # HubertModel has no freeze_feature_encoder(); this is what that
            # method calls in the models that have it.
            encoder.feature_extractor._freeze_parameters()

    def output_size(self) -> int:
        """Return the width of one output frame."""
        return int(self.encoder.config.hidden_size)

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of waveforms.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.

        Returns:
            tuple: Frames of shape ``(B, T, D)`` and the valid frame count per
            waveform.
        """
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


class W2vBertFrontend(AbsFrontend):
    """Hugging Face w2v-BERT 2.0 encoder on log-mel filterbank features.

    w2v-BERT 2.0 has no convolutional feature encoder. Its feature extractor
    computes 80-dimensional Kaldi-style filterbanks, normalises them per
    utterance and stacks every two frames into 160 dimensions, which is the
    input the encoder was pre-trained on.

    Args:
        encoder: ``transformers.Wav2Vec2BertModel``.
        feature_extractor: ``transformers.SeamlessM4TFeatureExtractor`` of the
            same checkpoint.
        sampling_rate: Sampling rate of the waveforms.

    Examples:
        .. code-block:: yaml

            frontend:
              _target_: espnet3.systems.cls.frontend.W2vBertFrontend
              encoder:
                _target_: transformers.AutoModel.from_pretrained
                pretrained_model_name_or_path: facebook/w2v-bert-2.0
              feature_extractor:
                _target_: transformers.AutoFeatureExtractor.from_pretrained
                pretrained_model_name_or_path: facebook/w2v-bert-2.0
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_extractor,
        sampling_rate: int = 16000,
    ) -> None:
        """Keep the encoder and its feature extractor."""
        super().__init__()
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def output_size(self) -> int:
        """Return the width of one output frame."""
        return int(self.encoder.config.hidden_size)

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of waveforms.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.

        Returns:
            tuple: Frames of shape ``(B, T, D)`` and the valid frame count per
            waveform.
        """
        features = self.feature_extractor(
            _split_batch(speech, speech_lengths),
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = features["input_features"].to(speech.device)
        attention_mask = features["attention_mask"].to(speech.device)
        frames = self.encoder(
            input_features, attention_mask=attention_mask
        ).last_hidden_state
        return frames, attention_mask.sum(dim=-1).to(torch.long)


class WhisperFrontend(AbsFrontend):
    """Encoder of a Hugging Face Whisper model on log-mel features.

    Whisper's encoder only accepts 30 seconds of input, so every waveform is
    zero-padded (or truncated) to 30 seconds by Whisper's own feature
    extractor, as in its pre-training. The returned lengths cover the frames
    of the real waveform only, so the padding never reaches the pooling.

    Args:
        whisper: ``transformers.WhisperModel``. Only its encoder is kept.
        feature_extractor: ``transformers.WhisperFeatureExtractor`` of the
            same checkpoint.
        sampling_rate: Sampling rate of the waveforms.
        freeze_subsampling: Whether to freeze the two convolutions in front of
            the transformer layers.

    Examples:
        .. code-block:: yaml

            frontend:
              _target_: espnet3.systems.cls.frontend.WhisperFrontend
              whisper:
                _target_: transformers.WhisperModel.from_pretrained
                pretrained_model_name_or_path: openai/whisper-medium
              feature_extractor:
                _target_: transformers.WhisperFeatureExtractor.from_pretrained
                pretrained_model_name_or_path: openai/whisper-medium
    """

    def __init__(
        self,
        whisper: nn.Module,
        feature_extractor,
        sampling_rate: int = 16000,
        freeze_subsampling: bool = True,
    ) -> None:
        """Keep the encoder and drop the decoder."""
        super().__init__()
        self.encoder = whisper.get_encoder()
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate
        # Whisper's positional embeddings are fixed sinusoids. Transformers
        # freezes them at construction, but from_pretrained hands them back
        # trainable, so they are frozen again here.
        self.encoder.embed_positions.requires_grad_(False)
        if freeze_subsampling:
            for conv in (self.encoder.conv1, self.encoder.conv2):
                conv.requires_grad_(False)

    def output_size(self) -> int:
        """Return the width of one output frame."""
        return int(self.encoder.config.d_model)

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of waveforms.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.

        Returns:
            tuple: Frames of shape ``(B, 1500, D)`` and the number of frames
            that cover the real waveform.
        """
        extractor = self.feature_extractor
        features = extractor(
            _split_batch(speech, speech_lengths),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            device=str(speech.device),
        )
        input_features = features["input_features"].to(speech.device)
        frames = self.encoder(input_features).last_hidden_state

        # The extractor makes one mel frame per hop and the second convolution
        # halves the frame count, rounding up.
        samples = torch.clamp(speech_lengths, max=extractor.n_samples)
        mel_frames = torch.div(samples, extractor.hop_length, rounding_mode="floor")
        frame_lengths = torch.div(mel_frames + 1, 2, rounding_mode="floor")
        return frames, torch.clamp(frame_lengths, min=1).to(torch.long)


def load_s2t_model(model_tag: str) -> nn.Module:
    """Download an ESPnet speech-to-text model (such as OWSM) and build it.

    Args:
        model_tag: Model name on the Hugging Face Hub, such as
            ``espnet/owsm_v4_medium_1B``.

    Returns:
        nn.Module: The ``ESPnetS2TModel`` with its pre-trained weights.
    """
    from espnet_model_zoo.downloader import ModelDownloader

    from espnet2.tasks.s2t import S2TTask

    files = ModelDownloader().download_and_unpack(model_tag)
    model, _ = S2TTask.build_model_from_file(
        files["s2t_train_config"], files["s2t_model_file"], device="cpu"
    )
    return model


class OwsmFrontend(AbsFrontend):
    """Encoder of an ESPnet speech-to-text model, such as OWSM, on log-mel.

    Keeps the model's own feature extraction, its global feature normalisation
    and its encoder, which is exactly how the encoder saw speech in
    pre-training. The decoder, the CTC layer and SpecAugment are dropped.

    Args:
        s2t_model: ``ESPnetS2TModel``, normally from :func:`load_s2t_model`.
        freeze_subsampling: Whether to freeze the convolutions of the
            subsampling in front of the encoder blocks. Its output projection
            stays trainable, like the projection after a wav2vec 2.0 feature
            encoder.
        layer_drop_rate: Probability of skipping each encoder block during
            training. ``None`` keeps the value the model was trained with.
        pad_seconds: When set, zero-pad (or truncate) every waveform to this
            many seconds and encode it at full length, as OWSM's own inference
            does with 30 seconds. The returned lengths still cover only the
            frames of the real waveform. ``None`` encodes each waveform at its
            own length, which is several times cheaper.
        sampling_rate: Sampling rate of the waveforms, used with
            ``pad_seconds``.

    Examples:
        .. code-block:: yaml

            frontend:
              _target_: espnet3.systems.cls.frontend.OwsmFrontend
              s2t_model:
                _target_: espnet3.systems.cls.frontend.load_s2t_model
                model_tag: espnet/owsm_v4_medium_1B
    """

    def __init__(
        self,
        s2t_model: nn.Module,
        freeze_subsampling: bool = True,
        layer_drop_rate: Optional[float] = None,
        pad_seconds: Optional[float] = None,
        sampling_rate: int = 16000,
    ) -> None:
        """Keep the feature extraction, the normalisation and the encoder."""
        super().__init__()
        self.feature = s2t_model.frontend
        self.normalize = s2t_model.normalize
        self.encoder = s2t_model.encoder
        if freeze_subsampling:
            self.encoder.embed.conv.requires_grad_(False)
        if layer_drop_rate is not None:
            # The encoder reads this attribute on every forward pass.
            self.encoder.layer_drop_rate = layer_drop_rate

        self.pad_samples = None
        if pad_seconds is not None:
            if type(self.encoder.embed).__name__ != "Conv2dSubsampling8":
                raise ValueError(
                    "pad_seconds counts the real frames for Conv2dSubsampling8 "
                    f"only, but the encoder uses {type(self.encoder.embed).__name__}"
                )
            self.pad_samples = int(pad_seconds * sampling_rate)

    def output_size(self) -> int:
        """Return the width of one output frame."""
        return int(self.encoder.output_size())

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of waveforms.

        Args:
            speech: Waveforms of shape ``(B, samples)``.
            speech_lengths: Valid sample count per waveform, shape ``(B,)``.

        Returns:
            tuple: Frames of shape ``(B, T, D)`` and the valid frame count per
            waveform.
        """
        real_lengths = speech_lengths
        if self.pad_samples is not None:
            real_lengths = torch.clamp(speech_lengths, max=self.pad_samples)
            speech = speech[:, : self.pad_samples]
            speech = F.pad(speech, (0, self.pad_samples - speech.size(1)))
            speech_lengths = torch.full_like(speech_lengths, self.pad_samples)

        feats, feats_lengths = self.feature(speech, speech_lengths)
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)
        frames, frame_lengths, _ = self.encoder(feats, feats_lengths)

        if self.pad_samples is not None:
            _, real_feats_lengths = self.feature(speech, real_lengths)
            frame_lengths = real_feats_lengths
            # Conv2dSubsampling8: three time-axis convolutions, kernel 3,
            # stride 2, counted per utterance as ESPnet does.
            for _ in range(3):
                frame_lengths = (
                    torch.div(frame_lengths - 3, 2, rounding_mode="floor") + 1
                )
        return frames, frame_lengths.to(torch.long)
