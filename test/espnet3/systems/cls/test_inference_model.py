"""Tests for the cls system's inference model."""

import pytest
import torch
from torch import nn

from espnet3.systems.cls.src.inference_model import ClsInferenceModel
from espnet3.systems.cls.src.model import ClsModel

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_loads_a_plain_state_dict               | Averaged *.pth weights load  |
# |                          | even though they carry the loss weights.        |
# | test_loads_a_lightning_checkpoint           | Per-epoch checkpoints keep   |
# |                                             | weights under state_dict.    |
# | test_labels_one_waveform                    | Returns (index, probs, name) |
# |                                             | in token_list order.         |
# | test_strips_the_lightning_model_prefix      | Keys saved from the Lightning|
# |                          | module (`model.` prefix) load into the network. |
# | test_labels_a_batch_like_one_by_one         | A list gives the same outputs|
# |                          | as one call per waveform, in input order.       |
# | test_sub_batches_stay_under_the_sample_cap  | Longest first; padded size   |
# |                          | of a group never exceeds max_batch_samples.     |
# | test_rejects_an_unknown_autocast            | Only bf16, fp16 or None.     |


class TinyFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 4)

    def output_size(self):
        return 4

    def forward(self, speech, speech_lengths):
        frames = speech[:, : (speech.size(1) // 160) * 160]
        frames = frames.reshape(speech.size(0), -1, 160).mean(-1, keepdim=True)
        return self.projection(frames), torch.div(
            speech_lengths, 160, rounding_mode="floor"
        )


def _network(**kwargs):
    torch.manual_seed(0)
    return ClsModel(TinyFrontend(), num_classes=3, head_size=8, **kwargs)


@pytest.fixture
def token_list(tmp_path):
    path = tmp_path / "token_list"
    path.write_text("angry\nhappy\nsad\n<unk>\n")
    return path


def _trained_state():
    return _network(class_weights=[1.0, 2.0, 3.0]).state_dict()


def test_loads_a_plain_state_dict(tmp_path, token_list):
    path = tmp_path / "model.pth"
    state = _trained_state()
    torch.save(state, path)

    model = ClsInferenceModel(_network(), path, token_list)

    assert torch.equal(model.model.head.output.weight, state["head.output.weight"])


def test_loads_a_lightning_checkpoint(tmp_path, token_list):
    path = tmp_path / "epoch000.ckpt"
    torch.save({"epoch": 0, "state_dict": _trained_state()}, path)

    model = ClsInferenceModel(_network(), path, token_list)

    assert not model.model.training


def test_labels_one_waveform(tmp_path, token_list):
    path = tmp_path / "model.pth"
    torch.save(_trained_state(), path)
    model = ClsInferenceModel(_network(), path, token_list)

    prediction, scores, label = model(torch.randn(16000).numpy())

    assert scores.shape == (3,)
    assert float(scores.sum()) == pytest.approx(1.0)
    assert label == ["angry", "happy", "sad"][prediction[0]]


def test_strips_the_lightning_model_prefix(tmp_path, token_list):
    path = tmp_path / "epoch000.ckpt"
    state = _trained_state()
    prefixed = {f"model.{key}": value for key, value in state.items()}
    torch.save({"epoch": 0, "state_dict": prefixed}, path)

    model = ClsInferenceModel(_network(), path, token_list)

    assert torch.equal(model.model.head.output.weight, state["head.output.weight"])


def test_labels_a_batch_like_one_by_one(tmp_path, token_list):
    path = tmp_path / "model.pth"
    torch.save(_trained_state(), path)
    # A small cap splits the four waveforms into several sub-batches.
    model = ClsInferenceModel(_network(), path, token_list, max_batch_samples=4000)
    torch.manual_seed(1)
    waves = [torch.randn(n).numpy() for n in (1600, 3200, 800, 2400)]

    batch = model(waves)
    single = [model(wave) for wave in waves]

    assert [label for _, _, label in batch] == [label for _, _, label in single]
    for (_, batch_scores, _), (_, single_scores, _) in zip(batch, single):
        assert torch.allclose(batch_scores, single_scores, atol=1e-5)


def test_sub_batches_stay_under_the_sample_cap(tmp_path, token_list):
    path = tmp_path / "model.pth"
    torch.save(_trained_state(), path)
    model = ClsInferenceModel(_network(), path, token_list, max_batch_samples=1000)
    lengths = [100, 500, 300, 250, 2000]

    groups = model._sub_batches(lengths)

    assert sorted(i for group in groups for i in group) == list(range(5))
    assert groups[0] == [4]  # longer than the cap: alone
    for group in groups:
        longest = max(lengths[i] for i in group)
        assert len(group) == 1 or len(group) * longest <= 1000


def test_rejects_an_unknown_autocast(tmp_path, token_list):
    path = tmp_path / "model.pth"
    torch.save(_trained_state(), path)

    with pytest.raises(ValueError, match="autocast"):
        ClsInferenceModel(_network(), path, token_list, autocast="int8")
