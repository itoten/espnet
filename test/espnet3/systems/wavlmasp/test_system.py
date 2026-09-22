"""Tests for the wavlmasp system stage hooks."""

from collections import Counter
from pathlib import Path

import pytest

from espnet3.systems.wavlmasp.system import WavlmAspSystem

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_class_weights_are_inverse_frequency    | w_c = N / (C * n_c), the     |
# |                          | definition sklearn calls "balanced".            |
# | test_class_weights_skip_added_symbols       | <unk> carries no count, so   |
# |                          | the file lines up with the predicted classes.   |
# | test_class_weights_follow_token_list_order  | The order matches the token  |
# |                          | list, not the counter's insertion order.        |
# | test_balanced_classes_all_weigh_one         | Equal counts leave every     |
# |                                             | weight at 1.


def _weights(tmp_path: Path, counter: Counter, labels: list[str]) -> list[float]:
    output = tmp_path / "class_weights"
    WavlmAspSystem._write_class_weights(counter, labels, output)
    return [float(line) for line in output.read_text().splitlines()]


def test_class_weights_are_inverse_frequency(tmp_path: Path):
    counter = Counter({"neutral": 60, "anger": 30, "fear": 10})
    labels = ["neutral", "anger", "fear", "<unk>"]

    weights = _weights(tmp_path, counter, labels)

    total, classes = 100, 3
    expected = [total / (classes * n) for n in (60, 30, 10)]
    assert weights == pytest.approx(expected, rel=1e-9)


def test_class_weights_skip_added_symbols(tmp_path: Path):
    counter = Counter({"neutral": 2, "anger": 2})
    labels = ["neutral", "anger", "<unk>"]

    assert len(_weights(tmp_path, counter, labels)) == 2


def test_class_weights_follow_token_list_order(tmp_path: Path):
    """The rarest class weighs most, wherever the counter happened to put it."""
    counter = Counter({"anger": 10, "neutral": 90})
    labels = ["neutral", "anger", "<unk>"]

    weights = _weights(tmp_path, counter, labels)

    assert weights[0] < weights[1]


def test_balanced_classes_all_weigh_one(tmp_path: Path):
    counter = Counter({"a": 5, "b": 5, "c": 5})
    labels = ["a", "b", "c", "<unk>"]

    assert _weights(tmp_path, counter, labels) == pytest.approx([1.0, 1.0, 1.0])
