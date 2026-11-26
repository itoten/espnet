"""Tests for the cls recipe's output function, one item or a batch."""

import torch

from egs3.multidataset_ser_03.cls.src.inference import build_output

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                       | Description                            |
# |---------------------------------|----------------------------------------|
# | test_one_item                   | One record with utt_id, hyp, ref,      |
# |                                 | score.                                 |
# | test_batch_gives_one_per_item   | Lists in, a list of aligned records    |
# |                                 | out.                                   |


def _output(label, scores):
    return [0], torch.tensor(scores), label


def test_one_item():
    record = build_output(
        {"utt_id": "u1", "label": "sad"}, _output("happy", [0.25, 0.75]), 0
    )

    assert record == {"utt_id": "u1", "hyp": "happy", "ref": "sad", "score": "0.25 0.75"}


def test_batch_gives_one_per_item():
    records = build_output(
        [{"utt_id": "u1", "label": "sad"}, {"label": "angry"}],
        [_output("happy", [1.0, 0.0]), _output("angry", [0.0, 1.0])],
        [3, 4],
    )

    assert [r["utt_id"] for r in records] == ["u1", "4"]
    assert [r["hyp"] for r in records] == ["happy", "angry"]
    assert [r["ref"] for r in records] == ["sad", "angry"]
