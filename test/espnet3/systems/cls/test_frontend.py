"""Tests for the cls system's frontends."""

import pytest
import torch
from torch import nn
from transformers import SeamlessM4TFeatureExtractor, WhisperFeatureExtractor

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling8,
)
from espnet3.systems.cls.frontend import (
    OwsmFrontend,
    W2vBertFrontend,
    Wav2Vec2Frontend,
    WhisperFrontend,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# Wav2Vec2Frontend
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_wav2vec2_frame_lengths_come_from_the_encoder | The encoder, not a     |
# |                          | formula, says how many frames a waveform became. |
# | test_wav2vec2_masks_the_padding             | The encoder is told the true |
# |                                             | waveform lengths.            |
# | test_wav2vec2_freezes_the_feature_encoder   | Only the convolutional front |
# |                                             | is frozen by default.        |
# | test_wav2vec2_freeze_can_be_turned_off      | Everything stays trainable.  |
# | test_wav2vec2_normalizes_before_encoding    | The waveform reaches the     |
# |                          | encoder already normalized.                     |
#
# W2vBertFrontend
# | test_w2v_bert_feeds_stacked_filterbanks     | 160-dim stacked filterbanks, |
# |                          | 49 frames per second, per-utterance lengths.    |
#
# WhisperFrontend
# | test_whisper_pads_to_thirty_seconds         | The encoder always sees 3000 |
# |                                             | mel frames.                  |
# | test_whisper_lengths_cover_the_real_waveform | 50 frames per second, capped|
# |                                             | at 1500 (30 s).              |
# | test_whisper_freezes_only_the_convolutions  | conv1/conv2 and the fixed    |
# |                          | positional embeddings frozen, layers trainable. |
#
# OwsmFrontend
# | test_owsm_runs_feature_normalize_encoder    | The S2T model's own feature  |
# |                          | extraction and normalization feed its encoder.  |
# | test_owsm_freezes_only_the_subsampling_convolutions | embed.conv frozen; |
# |                          | its projection and the blocks trainable.        |
# | test_owsm_padding_keeps_the_real_frame_count | Padded to a fixed length, |
# |                          | the frames of the real waveform are still found.|

HIDDEN = 16
CONV_KERNEL, CONV_STRIDE = 400, 320


class _Config:
    hidden_size = HIDDEN
    d_model = HIDDEN


class _FeatureEncoder(nn.Module):
    """The convolutional front of a wav2vec 2.0-family model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)

    def _freeze_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = False


class FakeWav2Vec2(nn.Module):
    """Stands in for a Hugging Face wav2vec 2.0-family model."""

    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.feature_extractor = _FeatureEncoder()
        self.projection = nn.Linear(1, HIDDEN)
        self.seen = {}

    def _get_feat_extract_output_lengths(self, input_lengths):
        return (
            torch.div(input_lengths - CONV_KERNEL, CONV_STRIDE, rounding_mode="floor")
            + 1
        )

    def forward(self, speech, attention_mask=None):
        self.seen["speech"] = speech
        self.seen["attention_mask"] = attention_mask
        frames = int(
            self._get_feat_extract_output_lengths(torch.tensor(speech.size(1)))
        )
        hidden = self.projection(speech[:, :frames].unsqueeze(-1))
        return type("Output", (), {"last_hidden_state": hidden})()


class FakeW2vBert(nn.Module):
    """Stands in for Wav2Vec2BertModel: one output frame per input frame."""

    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.projection = nn.Linear(160, HIDDEN)
        self.seen = {}

    def forward(self, input_features, attention_mask=None):
        self.seen["input_features"] = input_features
        hidden = self.projection(input_features)
        return type("Output", (), {"last_hidden_state": hidden})()


class FakeWhisperEncoder(nn.Module):
    """Stands in for WhisperEncoder, which only accepts 3000 mel frames."""

    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.conv1 = nn.Conv1d(80, HIDDEN, 3, padding=1)
        self.conv2 = nn.Conv1d(HIDDEN, HIDDEN, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(1500, HIDDEN)
        self.layers = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, input_features):
        if input_features.shape[-1] != 3000:
            raise ValueError("Whisper expects 3000 mel frames")
        hidden = self.conv2(self.conv1(input_features)).transpose(1, 2)
        return type("Output", (), {"last_hidden_state": self.layers(hidden)})()


class FakeWhisper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeWhisperEncoder()
        self.decoder = nn.Linear(1, 1)

    def get_encoder(self):
        return self.encoder


class FakeS2T(nn.Module):
    """Stands in for ESPnetS2TModel: feature extraction, normalization, encoder."""

    class Feature(nn.Module):
        def forward(self, speech, lengths):
            frames = speech.size(1) // 160
            feats = speech[:, : frames * 160].reshape(speech.size(0), frames, 160)
            return feats[..., :4], torch.div(lengths, 160, rounding_mode="floor")

    class Normalize(nn.Module):
        def forward(self, feats, lengths):
            return feats * 2, lengths

    class Encoder(nn.Module):
        class Subsampling(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Linear(4, HIDDEN)
                self.out = nn.Linear(HIDDEN, HIDDEN)

            def forward(self, feats):
                return self.out(self.conv(feats))

        def __init__(self):
            super().__init__()
            self.embed = self.Subsampling()
            self.blocks = nn.Linear(HIDDEN, HIDDEN)
            self.seen = {}

        def output_size(self):
            return HIDDEN

        def forward(self, feats, lengths):
            self.seen["feats"] = feats
            return self.blocks(self.embed(feats)), lengths, None

    def __init__(self):
        super().__init__()
        self.frontend = self.Feature()
        self.normalize = self.Normalize()
        self.encoder = self.Encoder()
        self.decoder = nn.Linear(1, 1)


def _speech(lengths, total=16000):
    speech = torch.randn(len(lengths), total) * 0.1
    return speech, torch.tensor(lengths)


def test_wav2vec2_frame_lengths_come_from_the_encoder():
    frontend = Wav2Vec2Frontend(FakeWav2Vec2())
    speech, lengths = _speech([16000, 8000])

    frames, frame_lengths = frontend(speech, lengths)

    assert frame_lengths.tolist() == [49, 24]
    assert frames.shape == (2, 49, HIDDEN)
    assert frontend.output_size() == HIDDEN


def test_wav2vec2_masks_the_padding():
    encoder = FakeWav2Vec2()
    speech, lengths = _speech([16000, 8000])

    Wav2Vec2Frontend(encoder)(speech, lengths)

    assert encoder.seen["attention_mask"].sum(dim=1).tolist() == [16000, 8000]


def test_wav2vec2_freezes_the_feature_encoder():
    encoder = FakeWav2Vec2()
    Wav2Vec2Frontend(encoder)
    assert not any(p.requires_grad for p in encoder.feature_extractor.parameters())
    assert all(p.requires_grad for p in encoder.projection.parameters())


def test_wav2vec2_freeze_can_be_turned_off():
    encoder = FakeWav2Vec2()
    Wav2Vec2Frontend(encoder, freeze_subsampling=False)
    assert all(p.requires_grad for p in encoder.parameters())


def test_wav2vec2_normalizes_before_encoding():
    class Doubler(nn.Module):
        def forward(self, speech, lengths):
            return speech * 2, lengths

    encoder = FakeWav2Vec2()
    speech, lengths = _speech([16000])

    Wav2Vec2Frontend(encoder, normalize=Doubler())(speech, lengths)

    assert torch.allclose(encoder.seen["speech"], speech * 2)


def test_w2v_bert_feeds_stacked_filterbanks():
    encoder = FakeW2vBert()
    frontend = W2vBertFrontend(encoder, SeamlessM4TFeatureExtractor())
    speech, lengths = _speech([16000, 8000])

    frames, frame_lengths = frontend(speech, lengths)

    assert encoder.seen["input_features"].shape == (2, 49, 160)
    assert frame_lengths.tolist() == [49, 24]
    assert frames.shape == (2, 49, HIDDEN)


def test_whisper_pads_to_thirty_seconds():
    frontend = WhisperFrontend(FakeWhisper(), WhisperFeatureExtractor())
    speech, lengths = _speech([16000, 8000])

    frames, _ = frontend(speech, lengths)

    assert frames.shape == (2, 1500, HIDDEN)


@pytest.mark.execution_timeout(30)
def test_whisper_lengths_cover_the_real_waveform():
    frontend = WhisperFrontend(FakeWhisper(), WhisperFeatureExtractor())
    speech, lengths = _speech([16000, 8000, 600000], total=600000)

    _, frame_lengths = frontend(speech, lengths)

    # 100 mel frames per second, halved by the stride-2 convolution; input
    # beyond 30 seconds is truncated by the extractor.
    assert frame_lengths.tolist() == [50, 25, 1500]


def test_whisper_freezes_only_the_convolutions():
    frontend = WhisperFrontend(FakeWhisper(), WhisperFeatureExtractor())
    encoder = frontend.encoder
    assert not any(p.requires_grad for p in encoder.conv1.parameters())
    assert not any(p.requires_grad for p in encoder.conv2.parameters())
    assert not encoder.embed_positions.weight.requires_grad
    assert all(p.requires_grad for p in encoder.layers.parameters())
    assert not hasattr(frontend, "decoder")


def test_owsm_runs_feature_normalize_encoder():
    s2t = FakeS2T()
    frontend = OwsmFrontend(s2t)
    speech, lengths = _speech([16000, 8000])

    frames, frame_lengths = frontend(speech, lengths)

    expected_feats = speech.reshape(2, 100, 160)[..., :4] * 2
    assert torch.allclose(s2t.encoder.seen["feats"], expected_feats)
    assert frame_lengths.tolist() == [100, 50]
    assert frames.shape == (2, 100, HIDDEN)
    assert frontend.output_size() == HIDDEN


def test_owsm_freezes_only_the_subsampling_convolutions():
    frontend = OwsmFrontend(FakeS2T())
    embed = frontend.encoder.embed
    assert not any(p.requires_grad for p in embed.conv.parameters())
    assert all(p.requires_grad for p in embed.out.parameters())
    assert all(p.requires_grad for p in frontend.encoder.blocks.parameters())
    assert not hasattr(frontend, "decoder")


class FakeS2TWithSubsampling(FakeS2T):
    """FakeS2T whose encoder subsamples with ESPnet's real Conv2dSubsampling8."""

    class Feature(nn.Module):
        def forward(self, speech, lengths):
            frames = speech.size(1) // 160
            feats = speech[:, : frames * 160].reshape(speech.size(0), frames, 160)
            return feats[..., :32], torch.div(lengths, 160, rounding_mode="floor")

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = Conv2dSubsampling8(32, HIDDEN, 0.0)
            self.seen = {}

        def output_size(self):
            return HIDDEN

        def forward(self, feats, lengths):
            self.seen["lengths"] = lengths
            mask = (torch.arange(feats.size(1))[None, :] < lengths[:, None])[:, None]
            frames, mask = self.embed(feats, mask)
            return frames, mask.squeeze(1).sum(dim=-1), None


def test_owsm_padding_keeps_the_real_frame_count():
    s2t = FakeS2TWithSubsampling()
    speech, lengths = _speech([16000, 8000])

    _, plain_lengths = OwsmFrontend(s2t)(speech, lengths)
    frames, padded_lengths = OwsmFrontend(s2t, pad_seconds=2)(speech, lengths)

    assert s2t.encoder.seen["lengths"].tolist() == [200, 200]
    assert torch.equal(padded_lengths, plain_lengths)
    assert frames.size(1) > int(plain_lengths.max())
