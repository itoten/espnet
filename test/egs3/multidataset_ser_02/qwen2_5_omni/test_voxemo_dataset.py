"""Tests for the evaluation dataset of multidataset_ser_02/qwen2_5_omni."""

import numpy as np
import pytest

import egs3.multidataset_ser_02.qwen2_5_omni.dataset.dataset as module
from egs3.multidataset_ser_02.qwen2_5_omni.dataset.dataset import VoxEmoEvalDataset

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                  | Description                   |
# |--------------------------------------------|-------------------------------|
# | test_label_set_in_recipe_order             | Present labels, LABELS order. |
# | test_items_carry_id_and_label_set          | utt_id, label_set added.      |
# | test_subsample_is_seeded_and_ordered       | Same seed, same rows, sorted. |
# | test_label_set_comes_from_the_whole_corpus | Not from the subsample.       |
# | test_rejects_unknown_labels                | Outside LABELS -> error.      |


class FakeCorpus:
    rows = []

    def __init__(self, split):
        self.split = split

    @property
    def labels(self):
        return [label for _, label in self.rows]

    @property
    def utt_ids(self):
        return [utt for utt, _ in self.rows]

    def __getitem__(self, idx):
        return {"speech": np.zeros(4, dtype=np.float32), "label": self.rows[idx][1]}


@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.setattr(module, "MultidatasetSer02Dataset", FakeCorpus)
    monkeypatch.setattr(
        module,
        "LABELS",
        ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"],
    )

    def set_rows(labels):
        FakeCorpus.rows = [(f"u{i}", label) for i, label in enumerate(labels)]

    return set_rows


def test_label_set_in_recipe_order(corpus):
    corpus(["angry", "neutral", "angry", "sad"])

    assert VoxEmoEvalDataset("x").label_set == ["neutral", "sad", "angry"]


def test_items_carry_id_and_label_set(corpus):
    corpus(["angry", "neutral"])

    item = VoxEmoEvalDataset("x")[1]

    assert item["utt_id"] == "u1"
    assert item["label"] == "neutral"
    assert item["label_set"] == "neutral, angry"


def test_subsample_is_seeded_and_ordered(corpus):
    corpus(["sad"] * 50)

    a = VoxEmoEvalDataset("x", max_utterances=10, seed=3)
    b = VoxEmoEvalDataset("x", max_utterances=10, seed=3)

    assert len(a) == 10
    assert a.indices == b.indices == sorted(a.indices)
    assert len(VoxEmoEvalDataset("x", max_utterances=100)) == 50


def test_label_set_comes_from_the_whole_corpus(corpus):
    corpus(["sad"] * 99 + ["fear"])

    ds = VoxEmoEvalDataset("x", max_utterances=1, seed=0)

    assert ds.label_set == ["sad", "fear"]


def test_rejects_unknown_labels(corpus):
    corpus(["sad", "calm"])

    with pytest.raises(ValueError, match="calm"):
        VoxEmoEvalDataset("x")
