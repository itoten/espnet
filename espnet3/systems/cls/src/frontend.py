"""Speech encoders that turn waveforms into frame features for the classifier.

Every frontend takes a batch of waveforms with their lengths and returns frame
features with their lengths, the interface of ESPnet2's ``AbsFrontend``. Each
one feeds its encoder the input that encoder was pre-trained on, so only the
encoder differs between models built on different frontends.
``LogMelTransformerFrontend`` is the exception: a plain transformer on log-mel
features, built with random weights to be trained from scratch.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import math

import torch
import torch.nn.functional as F
from torch import nn

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend

__all__ = [
    "LogMelTransformerFrontend",
    "LogMelEncoderFrontend",
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


def _min_samples_for_frames(config, frames: int) -> int:
    """Return the fewest samples the convolutional front turns into ``frames``."""
    samples = frames
    for kernel, stride in zip(
        reversed(config.conv_kernel), reversed(config.conv_stride)
    ):
        samples = (samples - 1) * stride + kernel
    return samples


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
        compile_layers: Whether to compile each transformer layer with
            ``torch.compile`` (``dynamic=True``, since the batch length
            varies). This fuses the many small operations in a layer and
            makes training faster; the first steps are slower while the
            layers compile. Compiling layer by layer, not the whole encoder,
            keeps LayerDrop, which skips layers at random in Python.
        compile_encoder: Whether to compile the whole encoder instead
            (``dynamic=True``); ``torch.compile`` breaks the graph where
            LayerDrop branches. Not combined with ``compile_layers``.
        compile_mode: ``mode`` given to ``torch.compile`` (``None`` for its
            default; ``"max-autotune-no-cudagraphs"`` also tunes the matrix
            products, at a much longer compile time).
        compile_feature_encoder: Whether to compile the convolutional feature
            encoder with ``torch.compile`` (``dynamic=True``). It runs on the
            raw waveform, so its normalisations and activations move a lot of
            memory; compiling fuses them, which mostly pays off together
            with ``compile_layers``.
        multilayer_feature: Whether to give the classifier a learned weighted
            sum of all the hidden states (the output of the convolutional
            front and of every transformer layer) instead of the last one.
            The weighted sum is ESPnet's ``Featureizer``
            (``espnet2.asr.frontend.espnet_ssl``, the S3PRL featurizer that
            ``S3prlFrontend`` uses with ``multilayer_feature``): softmax
            weights over the layers, initialised equal.
        freeze_encoder: Whether to freeze the whole encoder and run it in
            evaluation mode without gradients, as in SUPERB: only the
            weighted sum (with ``multilayer_feature``), the pooling and the
            head learn. Evaluation mode also turns off the encoder's own
            time masking, dropout and LayerDrop.
        max_train_crop: In training, cut every waveform of a batch to the
            shortest one when that removes at most this fraction of the
            padded length (``0`` never cuts). A batch without padding needs
            no attention mask, and only then does SDPA attention take its
            fastest (flash) kernel. It suits length-sorted batches, whose
            waveforms differ little in length; a small value such as
            ``0.05`` drops almost no audio. Evaluation is never cut.

    Examples:
        .. code-block:: yaml

            frontend:
              _target_: espnet3.systems.cls.src.frontend.Wav2Vec2Frontend
              encoder:
                _target_: transformers.AutoModel.from_pretrained
                pretrained_model_name_or_path: microsoft/wavlm-large
    """

    def __init__(
        self,
        encoder: nn.Module,
        normalize: Optional[nn.Module] = None,
        freeze_subsampling: bool = True,
        compile_layers: bool = False,
        compile_encoder: bool = False,
        compile_mode: Optional[str] = None,
        compile_feature_encoder: bool = False,
        max_train_crop: float = 0.0,
        multilayer_feature: bool = False,
        freeze_encoder: bool = False,
    ) -> None:
        """Keep the encoder, optionally freeze its front and compile it."""
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        self.max_train_crop = float(max_train_crop)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            encoder.requires_grad_(False)
            encoder.eval()
        self.featurizer = None
        if multilayer_feature:
            from espnet2.asr.frontend.espnet_ssl import Featureizer

            # hidden_states: the convolutional front, then each layer.
            self.featurizer = Featureizer(encoder.config.num_hidden_layers + 1)
        if freeze_subsampling:
            # HubertModel has no freeze_feature_encoder(); this is what that
            # method calls in the models that have it.
            encoder.feature_extractor._freeze_parameters()
        if compile_layers and compile_encoder:
            raise ValueError("Set compile_layers or compile_encoder, not both.")
        # nn.Module.compile() compiles in place, so the parameter names and
        # the checkpoints stay the same as without compiling.
        if compile_layers:
            for layer in encoder.encoder.layers:
                layer.compile(dynamic=True, mode=compile_mode)
        if compile_encoder:
            encoder.compile(dynamic=True, mode=compile_mode)
        if compile_feature_encoder and not compile_encoder:
            encoder.feature_extractor.compile(dynamic=True, mode=compile_mode)

    def output_size(self) -> int:
        """Return the width of one output frame."""
        return int(self.encoder.config.hidden_size)

    def train(self, mode: bool = True) -> "Wav2Vec2Frontend":
        """Set the training mode, keeping a frozen encoder in evaluation mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

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
        if self.training and self.max_train_crop > 0:
            shortest = int(speech_lengths.min())
            if shortest >= (1 - self.max_train_crop) * speech.size(1):
                speech = speech[:, :shortest]
                speech_lengths = torch.full_like(speech_lengths, shortest)

        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        config = self.encoder.config
        if (
            self.encoder.training
            and getattr(config, "apply_spec_augment", False)
            and getattr(config, "mask_time_prob", 0) > 0
        ):
            # SpecAugment raises if the padded batch is shorter than one time
            # mask; the extra zeros are outside the attention mask.
            min_samples = _min_samples_for_frames(config, config.mask_time_length)
            if speech.size(1) < min_samples:
                speech = F.pad(speech, (0, min_samples - speech.size(1)))

        mask = (
            torch.arange(speech.size(1), device=speech.device)[None, :]
            < speech_lengths[:, None]
        ).long()
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.freeze_encoder
        ):
            if self.featurizer is not None:
                out = self.encoder(
                    speech, attention_mask=mask, output_hidden_states=True
                )
            else:
                out = self.encoder(speech, attention_mask=mask)
        if self.featurizer is not None:
            frames = self.featurizer(list(out.hidden_states))
        else:
            frames = out.last_hidden_state
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
              _target_: espnet3.systems.cls.src.frontend.W2vBertFrontend
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
              _target_: espnet3.systems.cls.src.frontend.WhisperFrontend
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
              _target_: espnet3.systems.cls.src.frontend.OwsmFrontend
              s2t_model:
                _target_: espnet3.systems.cls.src.frontend.load_s2t_model
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


def _sinusoidal_positions(
    length: int, dim: int, device: torch.device
) -> torch.Tensor:
    """Return the fixed sinusoidal position table of the original Transformer."""
    position = torch.arange(length, device=device, dtype=torch.float)[:, None]
    scale = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float)
        * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(length, dim, device=device)
    table[:, 0::2] = torch.sin(position * scale)
    table[:, 1::2] = torch.cos(position * scale)
    return table


class LogMelTransformerFrontend(AbsFrontend):
    """Plain transformer encoder on log-mel features, trained from scratch.

    The waveform becomes log-mel filterbank features (ESPnet2's
    ``DefaultFrontend``), optionally masked by SpecAugment in training, then
    normalised: by ``normalize`` when given (as ESPnet2's ASR model does, e.g.
    ``GlobalMVN`` with statistics of the training set), otherwise standardised
    per utterance and per mel bin over its own frames. Strided 1-D
    convolutions bring the frames to the encoder width (with strides
    ``(1, 2)`` and a 10 ms hop: one frame every 20 ms, as in HuBERT), fixed
    sinusoidal positions are added, and pre-LayerNorm transformer layers
    follow, closed by a final LayerNorm. The layers are PyTorch's
    ``nn.TransformerEncoderLayer``, whose attention runs on
    ``scaled_dot_product_attention``.

    With the defaults (24 layers, width 1024, 16 heads, FFN 4096) the encoder
    has the depth and width of HuBERT Large and about 306M parameters, so that
    a model trained from scratch can be compared with a pre-trained one of the
    same size.

    Args:
        fs: Sampling rate of the waveforms.
        n_fft: FFT size of the log-mel features.
        win_length: Window length in samples.
        hop_length: Hop length in samples.
        n_mels: Number of mel bins.
        d_model: Width of the encoder.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        ffn_dim: Width of the feed-forward block of each layer.
        dropout: Dropout of the input, the attention and the feed-forward
            blocks.
        conv_strides: Stride of each convolution (kernel 3) in front of the
            layers; their product is the frame subsampling.
        specaug: Optional SpecAugment module (``espnet2.asr.specaug.specaug.
            SpecAug``), applied to the log-mel features in training only.
        normalize: Optional ESPnet2 normalisation layer (``AbsNormalize``,
            e.g. ``espnet2.layers.global_mvn.GlobalMVN`` on the
            ``logmel_stats.npz`` that ``collect_stats`` writes), applied after
            SpecAugment as in ESPnet2's ASR model. Without it each utterance is
            standardised over its own frames; SpecAugment's zeroed frames then
            enter those statistics and shift the unmasked frames too, so give
            a normalisation with fixed statistics when SpecAugment is on.
        compile_layers: Whether to compile each layer with ``torch.compile``
            (``dynamic=True``), as in :class:`Wav2Vec2Frontend`.
        compile_mode: ``mode`` given to ``torch.compile``.
        max_train_crop: In training, cut every waveform of a batch to the
            shortest one when that removes at most this fraction of the padded
            length (``0`` never cuts). A batch without padding needs no
            attention mask and runs on the fastest attention kernel; see
            :class:`Wav2Vec2Frontend`.

    Examples:
        .. code-block:: yaml

            frontend:
              _target_: espnet3.systems.cls.src.frontend.LogMelTransformerFrontend
              num_layers: 24
              d_model: 1024
              specaug:
                _target_: espnet2.asr.specaug.specaug.SpecAug
                apply_time_warp: false
                time_mask_width_ratio_range: [0.0, 0.05]
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        d_model: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        ffn_dim: int = 4096,
        dropout: float = 0.1,
        conv_strides: Sequence[int] = (1, 2),
        specaug: Optional[nn.Module] = None,
        normalize: Optional[nn.Module] = None,
        compile_layers: bool = False,
        compile_mode: Optional[str] = None,
        max_train_crop: float = 0.0,
    ) -> None:
        """Build the feature extraction and the encoder with random weights."""
        super().__init__()
        self.feature = DefaultFrontend(
            fs=fs,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.specaug = specaug
        self.normalize = normalize
        self.d_model = int(d_model)
        self.conv_strides = tuple(int(stride) for stride in conv_strides)
        self.max_train_crop = float(max_train_crop)

        convs = []
        channels = int(n_mels)
        for stride in self.conv_strides:
            convs += [
                nn.Conv1d(channels, self.d_model, 3, stride=stride, padding=1),
                nn.GELU(),
            ]
            channels = self.d_model
        self.subsample = nn.Sequential(*convs)
        self.dropout = nn.Dropout(dropout)
        # One layer at a time: nn.TransformerEncoder deep-copies a single
        # layer, which would start every layer from the same weights.
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                self.d_model,
                num_heads,
                ffn_dim,
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        if compile_layers:
            for layer in self.layers:
                layer.compile(dynamic=True, mode=compile_mode)

    def output_size(self) -> int:
        """Return the width of one output frame."""
        return self.d_model

    def logmel(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the log-mel features before SpecAugment and normalisation.

        ``ClsModel.collect_feats`` hands them to ``collect_stats``, which
        writes their statistics (``logmel_stats.npz``) for ``GlobalMVN``.
        """
        with torch.autocast(device_type=speech.device.type, enabled=False):
            return self.feature(speech.float(), speech_lengths)

    def _input_features(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crop, log-mel, SpecAugment (training only) and normalise a batch."""
        if self.training and self.max_train_crop > 0:
            shortest = int(speech_lengths.min())
            if shortest >= (1 - self.max_train_crop) * speech.size(1):
                speech = speech[:, :shortest]
                speech_lengths = torch.full_like(speech_lengths, shortest)

        # The STFT, the log and the normalisation run in fp32 even under bf16
        # autocast.
        with torch.autocast(device_type=speech.device.type, enabled=False):
            feats, feat_lengths = self.feature(speech.float(), speech_lengths)
            if self.specaug is not None and self.training:
                feats, feat_lengths = self.specaug(feats, feat_lengths)
            if self.normalize is not None:
                feats, feat_lengths = self.normalize(feats, feat_lengths)
            else:
                feats = self._standardise(feats, feat_lengths)
        return feats, feat_lengths

    @staticmethod
    def _standardise(feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Standardise each utterance per mel bin over its own frames."""
        valid = (
            torch.arange(feats.size(1), device=feats.device)[None, :]
            < lengths[:, None]
        ).unsqueeze(-1).to(feats.dtype)
        count = valid.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (feats * valid).sum(dim=1, keepdim=True) / count
        variance = (((feats - mean) * valid) ** 2).sum(dim=1, keepdim=True) / count
        return (feats - mean) / torch.sqrt(variance + 1.0e-5) * valid

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
        feats, feat_lengths = self._input_features(speech, speech_lengths)

        frames = self.subsample(feats.transpose(1, 2)).transpose(1, 2)
        frame_lengths = feat_lengths
        for stride in self.conv_strides:
            frame_lengths = torch.div(frame_lengths - 1, stride, rounding_mode="floor") + 1
        positions = _sinusoidal_positions(frames.size(1), self.d_model, frames.device)
        frames = self.dropout(frames + positions.to(frames.dtype))

        padding = (
            torch.arange(frames.size(1), device=frames.device)[None, :]
            >= frame_lengths[:, None]
        )
        # Without padding, no mask: SDPA then takes its fastest kernel.
        mask = padding if bool(padding.any()) else None
        for layer in self.layers:
            frames = layer(frames, src_key_padding_mask=mask)
        return self.final_norm(frames), frame_lengths.to(torch.long)


class LogMelEncoderFrontend(LogMelTransformerFrontend):
    """An ESPnet2 encoder (e.g. Conformer, E-Branchformer) on log-mel features.

    The same input stage as :class:`LogMelTransformerFrontend` (log-mel
    features, optional SpecAugment in training, normalisation (``normalize``,
    or per-utterance standardisation without it) and strided 1-D convolutions to the encoder width), followed by an ESPnet2
    ``AbsEncoder`` instead of the plain transformer layers. The encoder's own
    input layer is not used (``input_layer: null``): it adds its positional
    encoding to the frames of the convolutions, so all the models trained from
    scratch share the input stage and differ only in their encoder blocks.

    Config alone cannot do this: ESPnet2 encoders take features and lengths,
    not waveforms, and return a 3-tuple, while ``ClsModel`` expects a frontend
    that maps waveforms to frames.

    Args:
        encoder: A partially instantiated ESPnet2 encoder (``_partial_: true``);
            it is called with ``input_size=d_model``. Give it
            ``output_size: d_model`` and ``input_layer: null``.
        compile_layers: Whether to compile each encoder block with
            ``torch.compile`` (``dynamic=True``).
        Other arguments: as in :class:`LogMelTransformerFrontend`.

    Examples:
        .. code-block:: yaml

            frontend:
              _target_: espnet3.systems.cls.src.frontend.LogMelEncoderFrontend
              d_model: 1024
              encoder:
                _target_: espnet2.asr.encoder.conformer_encoder.ConformerEncoder
                _partial_: true
                output_size: 1024
                attention_heads: 16
                linear_units: 4096
                num_blocks: 12
                input_layer: null
                use_sdpa: true
    """

    def __init__(
        self,
        encoder,
        fs: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        d_model: int = 1024,
        conv_strides: Sequence[int] = (1, 2),
        specaug: Optional[nn.Module] = None,
        normalize: Optional[nn.Module] = None,
        compile_layers: bool = False,
        compile_mode: Optional[str] = None,
        max_train_crop: float = 0.0,
    ) -> None:
        """Build the input stage and the encoder with random weights."""
        super().__init__(
            fs=fs,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            d_model=d_model,
            num_layers=0,
            conv_strides=conv_strides,
            specaug=specaug,
            normalize=normalize,
            compile_layers=False,
            max_train_crop=max_train_crop,
        )
        # Parts of the plain transformer that this frontend does not use.
        del self.layers, self.final_norm, self.dropout
        self.encoder = encoder(input_size=self.d_model)
        if self.encoder.output_size() != self.d_model:
            raise ValueError(
                f"encoder output_size {self.encoder.output_size()} "
                f"must equal d_model {self.d_model}"
            )
        # Plain (absolute-position) attention on SDPA can run on the flash
        # kernel when a batch has no padding, but ESPnet2 encoders always pass
        # a padding mask. Their attention layers skip it when told the batch
        # has none (set per batch in `forward`). Relative-position attention
        # adds a position term to the scores and cannot use flash anyway.
        self._sdpa_attention = [
            module
            for module in self.encoder.modules()
            if type(module).__name__ == "MultiHeadedAttention"
            and getattr(module, "use_sdpa", False)
        ]
        if compile_layers:
            for block in self.encoder.encoders:
                block.compile(dynamic=True, mode=compile_mode)

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of waveforms into ``(B, T, d_model)`` frames."""
        feats, feat_lengths = self._input_features(speech, speech_lengths)

        frames = self.subsample(feats.transpose(1, 2)).transpose(1, 2)
        frame_lengths = feat_lengths
        for stride in self.conv_strides:
            frame_lengths = (
                torch.div(frame_lengths - 1, stride, rounding_mode="floor") + 1
            )
        if self._sdpa_attention:
            no_padding = bool((frame_lengths == frames.size(1)).all())
            for module in self._sdpa_attention:
                module.skip_mask = no_padding
        frames, frame_lengths, _ = self.encoder(frames, frame_lengths)
        return frames, frame_lengths.to(torch.long)
