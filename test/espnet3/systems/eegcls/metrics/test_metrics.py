"""Tests for the eegcls metrics."""

from pathlib import Path

import pytest

from espnet3.systems.eegcls.metrics.macro_f1 import MacroF1
from espnet3.systems.eegcls.metrics.ua import UA
from espnet3.systems.eegcls.metrics.wa import WA

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                          | Description                         |
# |------------------------------------|-------------------------------------|
# | test_wa_is_plain_accuracy          | Share of matching labels, in %.     |
# | test_ua_is_mean_recall             | Each class counts equally.          |
# | test_macro_f1_is_mean_f1           | F1 per class, then averaged.        |
# | test_metrics_reject_unknown_label  | A label outside token_list is an    |
# |                                    | error.                              |


def _write(tmp_path: Path, refs: list[str], hyps: list[str]):
    data = {}
    for key, labels in (("ref", refs), ("hyp", hyps)):
        path = tmp_path / f"{key}.scp"
        path.write_text("".join(f"utt{i} {label}\n" for i, label in enumerate(labels)))
        data[key] = path
    token_list = tmp_path / "token_list"
    token_list.write_text("right\nleft\n<unk>\n")
    return data, token_list


def test_wa_is_plain_accuracy(tmp_path: Path):
    data, _ = _write(tmp_path, ["right"] * 3 + ["left"], ["right"] * 3 + ["right"])

    assert WA()(data, "test", tmp_path) == {"WA": 75.0}


def test_ua_is_mean_recall(tmp_path: Path):
    # recall: right 3/3, left 0/1 -> 50
    data, token_list = _write(
        tmp_path, ["right"] * 3 + ["left"], ["right"] * 3 + ["right"]
    )

    result = UA(token_list=str(token_list))(data, "test", tmp_path)

    assert result == {"UA": 50.0}


def test_macro_f1_is_mean_f1(tmp_path: Path):
    # right: p=3/4 r=1 f1=6/7; left: p=0 r=0 f1=0 -> 42.86
    data, token_list = _write(
        tmp_path, ["right"] * 3 + ["left"], ["right"] * 3 + ["right"]
    )

    result = MacroF1(token_list=str(token_list))(data, "test", tmp_path)

    assert result["MacroF1"] == pytest.approx(42.86, abs=0.01)


def test_metrics_reject_unknown_label(tmp_path: Path):
    data, token_list = _write(tmp_path, ["right", "rest"], ["right", "left"])

    with pytest.raises(ValueError):
        UA(token_list=str(token_list))(data, "test", tmp_path)
