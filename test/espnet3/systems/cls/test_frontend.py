"""Tests for the cls system's frontends."""

import pytest
import torch
from torch import nn
from transformers import (
    SeamlessM4TFeatureExtractor,
    HubertConfig,
    HubertModel,
    WavLMConfig,
    WavLMModel,
    WhisperFeatureExtractor,
)

from espnet2.legacy.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling8,
)
from espnet3.systems.cls.src.frontend import (
    LogMelEncoderFrontend,
    LogMelTransformerFrontend,
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
# | test_wav2vec2_trains_on_a_batch_shorter_than_a_time_mask | A real    |
# |                          | WavLM with SpecAugment trains on a 0.1 s batch. |
# | test_wav2vec2_does_not_pad_in_eval          | Short input is left as is    |
# |                                             | outside training.            |
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


def _tiny_wavlm():
    config = WavLMConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8,) * 7,
        num_conv_pos_embeddings=8,
        mask_time_prob=0.5,
        mask_time_length=10,
    )
    return WavLMModel(config)


def test_wav2vec2_trains_on_a_batch_shorter_than_a_time_mask():
    # 1,600 samples become 4 frames, fewer than the 10 a time mask needs.
    frontend = Wav2Vec2Frontend(_tiny_wavlm()).train()
    speech, lengths = _speech([1600, 800], total=1600)

    frames, frame_lengths = frontend(speech, lengths)
    frames.sum().backward()

    assert frame_lengths.tolist() == [4, 2]
    assert frames.size(1) >= 10


def test_wav2vec2_does_not_pad_in_eval():
    frontend = Wav2Vec2Frontend(_tiny_wavlm()).eval()
    speech, lengths = _speech([1600], total=1600)

    frames, _ = frontend(speech, lengths)

    assert frames.size(1) == 4


def _tiny_hubert(layers=2, layerdrop=0.0):
    config = HubertConfig(
        layerdrop=layerdrop,
        hidden_size=HIDDEN,
        num_hidden_layers=layers,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8,) * 7,
        num_conv_pos_embeddings=8,
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
        mask_time_prob=0.5,
        mask_time_length=2,
    )
    return HubertModel(config)


def test_wav2vec2_weighted_sum_over_all_hidden_states():
    frontend = Wav2Vec2Frontend(_tiny_hubert(layers=2), multilayer_feature=True).eval()
    speech, lengths = _speech([3200, 1600], total=3200)

    frames, frame_lengths = frontend(speech, lengths)

    # The convolutional front and two layers: three equally weighted states.
    assert frontend.featurizer.weights.shape == (3,)
    out = frontend.encoder(
        speech, attention_mask=torch.ones_like(speech, dtype=torch.long), output_hidden_states=True
    )
    assert len(out.hidden_states) == 3
    assert frames.shape == out.last_hidden_state.shape
    assert frame_lengths.tolist() == [9, 4]


def test_wav2vec2_weighted_sum_learns_the_layer_weights():
    frontend = Wav2Vec2Frontend(_tiny_hubert(), multilayer_feature=True).train()
    speech, lengths = _speech([3200], total=3200)

    frames, _ = frontend(speech, lengths)
    frames.sum().backward()

    assert frontend.featurizer.weights.grad is not None
    assert any(p.grad is not None for p in frontend.encoder.encoder.parameters())


def test_wav2vec2_weighted_sum_needs_layerdrop_off_in_training():
    # transformers records hidden states from the layers that ran: with every
    # layer dropped the weighted sum gets one state, not three. The recipe
    # turns LayerDrop off for the weighted sum (hubert_large_nold_wsum.yaml).
    frontend = Wav2Vec2Frontend(
        _tiny_hubert(layers=2, layerdrop=1.0), multilayer_feature=True
    ).train()
    speech, lengths = _speech([3200], total=3200)
    with pytest.raises(RuntimeError):
        frontend(speech, lengths)


def test_wav2vec2_frozen_encoder_stays_in_eval_without_gradients():
    frontend = Wav2Vec2Frontend(
        _tiny_hubert(), multilayer_feature=True, freeze_encoder=True
    ).train()
    speech, lengths = _speech([3200], total=3200)

    assert frontend.training and not frontend.encoder.training
    frames, _ = frontend(speech, lengths)
    frames.sum().backward()

    assert frontend.featurizer.weights.grad is not None
    assert all(p.grad is None for p in frontend.encoder.parameters())
    assert not any(p.requires_grad for p in frontend.encoder.parameters())
    # Evaluation mode: no time masking, so the output is deterministic.
    again, _ = frontend(speech, lengths)
    assert torch.allclose(frames, again)


def test_wav2vec2_default_is_the_last_hidden_state():
    encoder = _tiny_hubert().eval()
    frontend = Wav2Vec2Frontend(encoder).eval()
    speech, lengths = _speech([3200], total=3200)

    frames, _ = frontend(speech, lengths)

    assert frontend.featurizer is None
    assert torch.allclose(frames, encoder(speech, attention_mask=torch.ones_like(speech, dtype=torch.long)).last_hidden_state)


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


def _two_padded_batches(frontend, short):
    """Encode `short` next to a 1 s and next to a 1.5 s waveform.

    The short waveform is zero-padded to the batch length in both batches, so
    its frames may differ only through how the frontend treats the padding.
    Returns [(frames, lengths) of batch A, (frames, lengths) of batch B].
    """
    out = []
    for total in (16000, 24000):
        batch = torch.zeros(2, total)
        batch[0] = torch.randn(total)
        batch[1, : short.numel()] = short
        lengths = torch.tensor([total, short.numel()])
        out.append(frontend(batch, lengths))
    return out


def _tiny_encoder(encoder_cls, **kwargs):
    from functools import partial

    return partial(
        encoder_cls,
        output_size=16,
        attention_heads=2,
        input_layer=None,
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["conformer", "ebranchformer"])
def test_logmel_encoder_frontend_lengths_and_padding(kind):
    from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
    from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder

    if kind == "conformer":
        # The settings of conf/frontend/conformer.yaml: ESPnet2's default
        # rel_pos_type "legacy" makes frames depend on the padding.
        encoder = _tiny_encoder(
            ConformerEncoder,
            linear_units=32,
            num_blocks=2,
            rel_pos_type="latest",
            macaron_style=True,
            use_sdpa=True,
        )
    else:
        encoder = _tiny_encoder(
            EBranchformerEncoder,
            cgmlp_linear_units=32,
            linear_units=32,
            num_blocks=2,
            use_ffn=True,
            macaron_ffn=True,
            use_sdpa=True,
        )
    frontend = LogMelEncoderFrontend(encoder=encoder, d_model=16).eval()
    assert frontend.output_size() == 16

    short = torch.randn(8000)
    frames, frame_lengths = _two_padded_batches(frontend, short)[0]
    # one frame every 20 ms: 101 centred 10 ms frames, then stride 2
    assert frame_lengths.tolist() == [51, 26]
    assert frames.shape[0] == 2 and frames.shape[2] == 16
    assert frames.size(1) >= int(frame_lengths.max())

    # the amount of padding must not change a shorter utterance's frames
    (frames_a, _), (frames_b, _) = _two_padded_batches(frontend, short)[:2]
    torch.testing.assert_close(frames_a[1, :26], frames_b[1, :26], atol=1e-4, rtol=1e-4)


def test_logmel_encoder_frontend_rejects_a_wrong_output_width():
    from espnet2.asr.encoder.conformer_encoder import ConformerEncoder

    encoder = _tiny_encoder(
        ConformerEncoder, linear_units=32, num_blocks=1, rel_pos_type="latest"
    )
    with pytest.raises(ValueError, match="output_size"):
        LogMelEncoderFrontend(encoder=encoder, d_model=32)


def test_logmel_transformer_frontend_runs_and_pads_consistently():
    frontend = LogMelTransformerFrontend(
        d_model=16, num_layers=2, num_heads=2, ffn_dim=32
    ).eval()
    short = torch.randn(8000)
    (frames_a, lengths_a), (frames_b, _) = _two_padded_batches(frontend, short)
    assert lengths_a.tolist() == [51, 26]
    torch.testing.assert_close(frames_a[1, :26], frames_b[1, :26], atol=1e-4, rtol=1e-4)


def test_logmel_encoder_frontend_drops_the_mask_only_without_padding():
    from espnet2.asr.encoder.conformer_encoder import ConformerEncoder

    encoder = _tiny_encoder(
        ConformerEncoder,
        linear_units=32,
        num_blocks=2,
        rel_pos_type="latest",
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        use_flash_attn=False,
        use_sdpa=True,
    )
    frontend = LogMelEncoderFrontend(encoder=encoder, d_model=16).eval()
    attention = frontend._sdpa_attention
    assert len(attention) == 2

    speech = torch.randn(2, 16000)
    full = frontend(speech, torch.tensor([16000, 16000]))[0]
    assert all(module.skip_mask for module in attention)
    frontend(speech, torch.tensor([16000, 8000]))
    assert not any(module.skip_mask for module in attention)

    # Without padding, dropping the all-true mask changes nothing.
    frontend._sdpa_attention = []
    for module in attention:
        module.skip_mask = False
    masked = frontend(speech, torch.tensor([16000, 16000]))[0]
    torch.testing.assert_close(full, masked, atol=1e-5, rtol=1e-5)


def _logmel_stats_file(tmp_path, frontend, speech, lengths):
    """Write a GlobalMVN statistics file of the frontend's own log-mel features."""
    import numpy as np

    feats, feat_lengths = frontend.logmel(speech, lengths)
    frames = torch.cat([f[: int(n)] for f, n in zip(feats, feat_lengths)])
    path = tmp_path / "logmel_stats.npz"
    np.savez(
        path,
        count=frames.size(0),
        sum=frames.sum(0).numpy(),
        sum_square=(frames**2).sum(0).numpy(),
    )
    return path


def test_logmel_frontend_normalize_replaces_the_per_utterance_standardisation(
    tmp_path,
):
    from espnet2.layers.global_mvn import GlobalMVN

    torch.manual_seed(0)
    speech = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 12000])
    plain = LogMelTransformerFrontend(d_model=16, num_layers=1, num_heads=2, ffn_dim=32)
    stats = _logmel_stats_file(tmp_path, plain, speech, lengths)
    normalize = GlobalMVN(stats)
    frontend = LogMelTransformerFrontend(
        d_model=16, num_layers=1, num_heads=2, ffn_dim=32, normalize=normalize
    ).eval()

    feats, feat_lengths = frontend._input_features(speech, lengths)
    raw, _ = frontend.logmel(speech, lengths)
    expected, _ = GlobalMVN(stats)(raw.clone(), feat_lengths)
    torch.testing.assert_close(feats, expected)
    # without `normalize` the per-utterance standardisation stays as it was
    plain.eval()
    feats_plain, _ = plain._input_features(speech, lengths)
    torch.testing.assert_close(feats_plain, plain._standardise(raw, feat_lengths))


def test_logmel_frontend_specaug_leaves_unmasked_frames_alone_with_globalmvn(
    tmp_path,
):
    """Global statistics keep the unmasked frames as at inference."""
    from espnet2.asr.specaug.specaug import SpecAug
    from espnet2.layers.global_mvn import GlobalMVN

    torch.manual_seed(0)
    speech = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 16000])
    specaug = SpecAug(
        apply_time_warp=False,
        apply_freq_mask=False,
        apply_time_mask=True,
        time_mask_width_range=(20, 30),
        num_time_mask=2,
    )
    probe = LogMelTransformerFrontend(d_model=16, num_layers=1, num_heads=2, ffn_dim=32)
    stats = _logmel_stats_file(tmp_path, probe, speech, lengths)
    frontend = LogMelTransformerFrontend(
        d_model=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        specaug=specaug,
        normalize=GlobalMVN(stats),
    )
    frontend.train()
    masked, _ = frontend._input_features(speech, lengths)
    frontend.eval()
    clean, _ = frontend._input_features(speech, lengths)
    raw, _ = frontend.logmel(speech, lengths)
    # frames SpecAugment zeroed: the log-mel is 0 there, everywhere else the
    # training input equals the inference input
    zeroed = (masked == GlobalMVN(stats)(torch.zeros_like(raw), lengths * 0 + raw.size(1))[0])
    unmasked = ~zeroed.all(dim=-1)
    assert unmasked.any() and (~unmasked).any()
    torch.testing.assert_close(masked[unmasked], clean[unmasked])


def test_cls_model_collect_feats_gives_the_logmel_of_a_logmel_frontend():
    from espnet3.systems.cls.src.model import ClsModel

    frontend = LogMelTransformerFrontend(d_model=16, num_layers=1, num_heads=2, ffn_dim=32)
    model = ClsModel(frontend=frontend, num_classes=3, head_size=8).eval()
    speech = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 8000])
    feats = model.collect_feats(speech, lengths)
    assert set(feats) == {"feats", "feats_lengths", "logmel", "logmel_lengths"}
    assert feats["logmel"].shape[-1] == 80
    assert feats["logmel_lengths"].tolist() == [101, 51]
