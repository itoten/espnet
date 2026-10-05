"""Tests for the eegcls system stage hooks."""

from collections import Counter
from pathlib import Path

import pytest

from espnet3.systems.eegcls.system import EegClsSystem

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_class_weights_are_inverse_frequency    | w_c = N / (C * n_c).         |
# | test_class_weights_skip_added_symbols       | <unk> carries no weight.     |
# | test_class_weights_follow_token_list_order  | Order follows the token list.|
# | test_balanced_classes_all_weigh_one         | Equal counts give weight 1.  |


def _weights(tmp_path: Path, counter: Counter, labels: list[str]) -> list[float]:
    output = tmp_path / "class_weights"
    EegClsSystem._write_class_weights(counter, labels, output)
    return [float(line) for line in output.read_text().splitlines()]


def test_class_weights_are_inverse_frequency(tmp_path: Path):
    counter = Counter({"right": 60, "left": 30, "rest": 10})
    labels = ["right", "left", "rest", "<unk>"]

    weights = _weights(tmp_path, counter, labels)

    expected = [100 / (3 * n) for n in (60, 30, 10)]
    assert weights == pytest.approx(expected, rel=1e-9)


def test_class_weights_skip_added_symbols(tmp_path: Path):
    counter = Counter({"right": 2, "left": 2})

    assert len(_weights(tmp_path, counter, ["right", "left", "<unk>"])) == 2


def test_class_weights_follow_token_list_order(tmp_path: Path):
    counter = Counter({"left": 10, "right": 90})

    weights = _weights(tmp_path, counter, ["right", "left", "<unk>"])

    assert weights[0] < weights[1]


def test_balanced_classes_all_weigh_one(tmp_path: Path):
    counter = Counter({"a": 5, "b": 5, "c": 5})

    weights = _weights(tmp_path, counter, ["a", "b", "c", "<unk>"])

    assert weights == pytest.approx([1.0, 1.0, 1.0])
