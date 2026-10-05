"""Tests for the eegcls inference model."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from espnet3.systems.eegcls.inference_model import EegClsInferenceModel
from espnet3.systems.eegcls.model import EegClsModel

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_loads_weights_saved_from_eegclsmodel   | A state dict of EegClsModel  |
# |                                             | loads into a fresh network.  |
# | test_call_returns_prediction_scores_label   | One (T, C) epoch gives the   |
# |                                             | argmax, probabilities, name. |


def _save(tmp_path: Path, time: int = 5, channels: int = 3):
    torch.manual_seed(0)
    trained = EegClsModel(nn.Sequential(nn.Flatten(), nn.Linear(time * channels, 2)))
    model_file = tmp_path / "model.pth"
    torch.save(trained.state_dict(), model_file)
    token_list = tmp_path / "token_list"
    token_list.write_text("right\nleft\n<unk>\n")
    return trained, model_file, token_list


def test_loads_weights_saved_from_eegclsmodel(tmp_path: Path):
    trained, model_file, token_list = _save(tmp_path)
    network = nn.Sequential(nn.Flatten(), nn.Linear(15, 2))

    model = EegClsInferenceModel(network, model_file, token_list)

    for key, value in trained.state_dict().items():
        assert torch.equal(model.model.state_dict()[key], value)


def test_call_returns_prediction_scores_label(tmp_path: Path):
    trained, model_file, token_list = _save(tmp_path)
    model = EegClsInferenceModel(
        nn.Sequential(nn.Flatten(), nn.Linear(15, 2)), model_file, token_list
    )
    eeg = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)

    prediction, scores, label = model(eeg)

    expected = trained.compute_logits(torch.from_numpy(eeg).unsqueeze(0))
    expected = torch.softmax(expected, dim=-1)[0]
    assert torch.allclose(scores, expected.detach())
    assert prediction == [int(expected.argmax())]
    assert label == ["right", "left"][prediction[0]]
