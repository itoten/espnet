"""Tests for the cls system's zero-shot Vox-Profile WavLM inference model.

A tiny WavLM encoder, layer weights and an emotion head are saved in the
released checkpoint's layout; the hub, the WavLM config and the feature
extractor are stubbed so nothing is downloaded.
"""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

import espnet3.systems.cls.src.vox_profile_wavlm_inference_model as module
from espnet3.systems.cls.src.vox_profile_inference_model import _EmotionHead
from espnet3.systems.cls.src.vox_profile_wavlm_inference_model import (
    BACKBONE_PREFIX,
    VoxProfileWavLMEmotionInferenceModel,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                          | Description                         |
# |------------------------------------|-------------------------------------|
# | test_reads_weights                 | Encoder, layer weights and emotion  |
# |                                    | head are read; other heads are not. |
# | test_matches_a_direct_computation  | Probabilities equal the softmax-    |
# |                                    | weighted sum of all hidden states   |
# |                                    | computed by hand.                   |
# | test_cuts_at_15_seconds            | Audio after 15 s is ignored.        |
# | test_batch_equals_one_by_one       | Each utterance runs on its own.     |
# | test_maps_and_renormalises_scores  | Contempt / Other dropped.           |
# | test_rejects_other_checkpoints     | Non-WavLM / LoRA configs fail.      |

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
    "finetune_method": "finetune",
    "freeze_params": True,
    "hidden_dim": 4,
    "lora_rank": 16,
    "output_class_num": 9,
    "pretrain_model": "wavlm_large",
    "use_conv_output": True,
}


def _tiny_config():
    return WavLMConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(8, 8),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
        num_feat_extract_layers=2,
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
        num_buckets=8,
        max_bucket_distance=16,
    )


@pytest.fixture
def released(tmp_path, monkeypatch):
    """Save a tiny checkpoint in the released layout and stub the hub."""
    torch.manual_seed(0)
    encoder = WavLMModel(_tiny_config()).eval()
    head = _EmotionHead(8, 4, 9).eval()
    layer_weights = torch.randn(3)

    state = {BACKBONE_PREFIX + k: v.clone() for k, v in encoder.state_dict().items()}
    state["weights"] = layer_weights.clone()
    state.update({k: v.clone() for k, v in head.state_dict().items()})
    state["arousal_layer.0.weight"] = torch.zeros(4, 4)
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(MODEL_CONFIG))

    monkeypatch.setattr(
        module, "hf_hub_download", lambda repo, name, revision=None: str(tmp_path / name)
    )
    monkeypatch.setattr(module.WavLMConfig, "from_pretrained", lambda name: _tiny_config())
    monkeypatch.setattr(
        module.Wav2Vec2FeatureExtractor,
        "from_pretrained",
        lambda name: Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, do_normalize=True, return_attention_mask=True
        ),
    )
    return dict(path=tmp_path, encoder=encoder, head=head, weights=layer_weights)


@pytest.fixture
def token_list(tmp_path):
    path = tmp_path / "token_list"
    path.write_text("angry\nhappy\nsad\nfear\nsurprise\ndisgust\nneutral\n<unk>\n")
    return path


def _model(token_list, **kwargs):
    kwargs.setdefault("label_map", LABEL_MAP)
    return VoxProfileWavLMEmotionInferenceModel(token_list=token_list, **kwargs)


def _waveforms():
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal(16000 * 2).astype(np.float32),
        rng.standard_normal(16000 * 20).astype(np.float32),
        rng.standard_normal(8000).astype(np.float32),
    ]


def test_reads_weights(released):
    encoder_state, weights, head_state = (
        VoxProfileWavLMEmotionInferenceModel._read_weights(
            str(released["path"] / "model.safetensors")
        )
    )
    assert set(encoder_state) == set(released["encoder"].state_dict())
    assert torch.equal(weights, released["weights"])
    assert set(head_state) == set(_EmotionHead(8, 4, 9).state_dict())


def _direct(released, waveform):
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, do_normalize=True
    )
    values = extractor(
        waveform[: 15 * 16000], sampling_rate=16000, return_tensors="pt"
    ).input_values
    with torch.no_grad():
        out = released["encoder"](values, output_hidden_states=True)
        # transformers 4.46 (the released code): the last state is the output
        hidden = list(out.hidden_states[:-1]) + [out.last_hidden_state]
        w = torch.softmax(released["weights"], dim=-1)
        frames = sum(w[i] * h for i, h in enumerate(hidden))
        feats = released["head"].model_seq(frames.transpose(1, 2)).transpose(1, 2)
        return torch.softmax(released["head"].emotion_layer(feats.mean(1)), dim=-1)


def test_matches_a_direct_computation(released, token_list):
    model = _model(token_list)
    waveforms = _waveforms()

    probabilities = model.label_probabilities(waveforms)

    expected = torch.cat([_direct(released, w) for w in waveforms])
    assert torch.allclose(probabilities, expected, atol=1e-5)


def test_cuts_at_15_seconds(released, token_list):
    model = _model(token_list)
    long = _waveforms()[1]

    full = model.label_probabilities([long])
    cut = model.label_probabilities([long[: 15 * 16000]])

    assert torch.allclose(full, cut, atol=1e-6)


def test_batch_equals_one_by_one(released, token_list):
    model = _model(token_list)
    waveforms = _waveforms()

    batch = model(waveforms)
    single = [model(torch.from_numpy(w)) for w in waveforms]

    for (b_pred, b_scores, b_label), (s_pred, s_scores, s_label) in zip(batch, single):
        assert b_pred == s_pred and b_label == s_label
        assert torch.allclose(b_scores, s_scores, atol=1e-6)


def test_maps_and_renormalises_scores(released, token_list):
    model = _model(token_list)
    waveform = _waveforms()[0]

    prediction, scores, label = model(waveform)

    row = model.label_probabilities([waveform])[0]
    kept = torch.stack([row[i] for i in (0, 4, 6, 3, 7, 2, 5)])  # token_list order
    assert torch.allclose(scores, kept / kept.sum(), atol=1e-6)
    assert label == model.classes[prediction[0]]


@pytest.mark.parametrize(
    "change", [{"pretrain_model": "whisper_large"}, {"finetune_method": "lora"}]
)
def test_rejects_other_checkpoints(released, token_list, change):
    (released["path"] / "config.json").write_text(json.dumps({**MODEL_CONFIG, **change}))
    with pytest.raises(ValueError, match="WavLM Large"):
        _model(token_list)
