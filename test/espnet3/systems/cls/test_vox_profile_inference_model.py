"""Tests for the cls system's zero-shot Vox-Profile inference model.

A tiny Whisper encoder with a LoRA layer and an emotion head is saved in the
released checkpoint's layout; the hub, the Whisper config and the feature
extractor are stubbed so nothing is downloaded.
"""

import copy
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

import espnet3.systems.cls.src.vox_profile_inference_model as module
from espnet3.systems.cls.src.vox_profile_inference_model import (
    ENCODER_PREFIX,
    VoxProfileEmotionInferenceModel,
    _EmotionHead,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                  | Description                   |
# |--------------------------------------------|-------------------------------|
# | test_frame_lengths                         | Released formula, clipped to  |
# |                                            | [1, 750].                     |
# | test_reads_weights_and_merges_lora         | W + B @ A / r; other heads    |
# |                                            | and the decoder are skipped.  |
# | test_matches_a_direct_computation          | Probabilities equal the LoRA  |
# |                                            | network computed by hand.     |
# | test_batch_equals_one_by_one               | Padding to 15 s makes batches |
# |                                            | exact.                        |
# | test_maps_and_renormalises_scores          | Contempt / Other dropped;     |
# |                                            | token_list order.             |
# | test_passes_revision_to_the_hub            | Pinned revision is used.      |
# | test_rejects_other_checkpoints             | Non-Whisper configs fail.     |
# | test_rejects_unknown_label                 | label_map typos fail early.   |

RANK = 2
LABEL_MAP = {
    "Anger": "angry",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
}
MODEL_CONFIG = {
    "detailed_class_num": 17,
    "finetune_method": "lora",
    "freeze_params": True,
    "hidden_dim": 4,
    "lora_rank": RANK,
    "output_class_num": 9,
    "pretrain_model": "whisper_large",
    "use_conv_output": True,
}


def _tiny_config():
    config = WhisperConfig(
        d_model=8,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        num_mel_bins=80,
        max_source_positions=750,
    )
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def released(tmp_path, monkeypatch):
    """Save a tiny checkpoint in the released layout and stub the hub."""
    torch.manual_seed(0)
    encoder = WhisperEncoder(_tiny_config()).eval()
    head = _EmotionHead(8, 4, 9).eval()
    lora_a = torch.randn(RANK, 8)
    lora_b = torch.randn(16, RANK)

    state = {ENCODER_PREFIX + k: v.clone() for k, v in encoder.state_dict().items()}
    state[ENCODER_PREFIX + "layers.1.fc1.lora_A"] = lora_a
    state[ENCODER_PREFIX + "layers.1.fc1.lora_B"] = lora_b
    state.update({k: v.clone() for k, v in head.state_dict().items()})
    state["arousal_layer.0.weight"] = torch.zeros(4, 4)
    state["backbone_model.decoder.layer_norm.weight"] = torch.zeros(8)
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(MODEL_CONFIG))

    calls = []

    def fake_download(repo, filename, revision=None):
        calls.append((repo, filename, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        module.WhisperConfig, "from_pretrained", lambda name: _tiny_config()
    )
    monkeypatch.setattr(
        module.WhisperFeatureExtractor,
        "from_pretrained",
        lambda name, chunk_length: WhisperFeatureExtractor(
            feature_size=80, chunk_length=chunk_length
        ),
    )

    merged = copy.deepcopy(encoder)
    merged.layers[1].fc1.weight.data += lora_b @ lora_a / RANK
    return dict(
        path=tmp_path, encoder=merged, head=head, calls=calls, lora=(lora_a, lora_b)
    )


@pytest.fixture
def token_list(tmp_path):
    path = tmp_path / "token_list"
    path.write_text("angry\nhappy\nsad\nfear\nsurprise\ndisgust\nneutral\n<unk>\n")
    return path


def _model(token_list, **kwargs):
    kwargs.setdefault("label_map", LABEL_MAP)
    return VoxProfileEmotionInferenceModel(
        token_list=token_list, attn_implementation="eager", **kwargs
    )


def _waveforms():
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal(16000 * 2).astype(np.float32),
        rng.standard_normal(16000 * 20).astype(np.float32),
        rng.standard_normal(8000).astype(np.float32),
    ]


def test_frame_lengths():
    lengths = VoxProfileEmotionInferenceModel.frame_lengths([0, 16000, 240000, 480000])

    assert lengths == [1, 50, 750, 750]


def test_reads_weights_and_merges_lora(released):
    encoder_state, head_state = VoxProfileEmotionInferenceModel._read_weights(
        str(released["path"] / "model.safetensors"), RANK, "lora"
    )

    lora_a, lora_b = released["lora"]
    expected = released["encoder"].layers[1].fc1.weight
    assert torch.allclose(encoder_state["layers.1.fc1.weight"], expected)
    assert not any("lora" in k or "decoder" in k for k in encoder_state)
    assert set(head_state) == set(_EmotionHead(8, 4, 9).state_dict())


def test_matches_a_direct_computation(released, token_list):
    model = _model(token_list)
    waveforms = _waveforms()

    probabilities = model.label_probabilities(waveforms)

    extractor = WhisperFeatureExtractor(feature_size=80, chunk_length=15)
    features = extractor(
        waveforms, sampling_rate=16000, return_tensors="pt", max_length=240000
    ).input_features
    with torch.no_grad():
        frames = released["encoder"](features).last_hidden_state
        hidden = released["head"].model_seq(frames.transpose(1, 2)).transpose(1, 2)
        pooled = torch.stack(
            [hidden[0, :100].mean(0), hidden[1, :750].mean(0), hidden[2, :25].mean(0)]
        )
        expected = torch.softmax(released["head"].emotion_layer(pooled), dim=-1)
    assert torch.allclose(probabilities, expected, atol=1e-5)


def test_batch_equals_one_by_one(released, token_list):
    model = _model(token_list)
    waveforms = _waveforms()

    batch = model(waveforms)
    single = [model(torch.from_numpy(w)) for w in waveforms]

    for (b_pred, b_scores, b_label), (s_pred, s_scores, s_label) in zip(batch, single):
        assert b_pred == s_pred and b_label == s_label
        assert torch.allclose(b_scores, s_scores, atol=1e-5)


def test_maps_and_renormalises_scores(released, token_list):
    model = _model(token_list)
    waveform = _waveforms()[0]

    prediction, scores, label = model(waveform)

    raw = model.label_probabilities([waveform])[0]
    order = ["Anger", "Happiness", "Sadness", "Fear", "Surprise", "Disgust", "Neutral"]
    expected = raw[[model.LABELS.index(name) for name in order]]
    assert torch.allclose(scores, expected / expected.sum())
    assert scores.shape == (7,)
    assert label == ["angry", "happy", "sad", "fear", "surprise", "disgust", "neutral"][
        prediction[0]
    ]


def test_passes_revision_to_the_hub(released, token_list):
    _model(token_list, model_tag="some/model", revision="abc123")

    assert ("some/model", "model.safetensors", "abc123") in released["calls"]
    assert ("some/model", "config.json", "abc123") in released["calls"]


def test_rejects_other_checkpoints(released, token_list):
    (released["path"] / "config.json").write_text(
        json.dumps({**MODEL_CONFIG, "pretrain_model": "wavlm_large"})
    )

    with pytest.raises(ValueError, match="Whisper Large"):
        _model(token_list)


def test_rejects_unknown_label(released, token_list):
    with pytest.raises(ValueError, match="labels the model does not have"):
        _model(token_list, label_map={"Angry": "angry"})
