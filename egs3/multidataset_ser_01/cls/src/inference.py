"""Output formatting for the cls recipe's inference stage."""

from __future__ import annotations

from typing import Sequence

__all__ = ["build_output"]


def build_output(data, model_output, idx):
    """Build one SCP record from a model output, or one per item of a batch.

    With ``batch_size`` set in the inference config, ``data``, ``idx`` and
    ``model_output`` are lists aligned item by item, and a list of records is
    returned.
    """
    if isinstance(idx, (list, tuple)):
        return [
            _record(item, output, i)
            for item, output, i in zip(data, model_output, idx)
        ]
    return _record(data, model_output, idx)


def _record(data: dict, model_output: Sequence, idx: int) -> dict:
    """Build the SCP record of one item."""
    _, scores, hyp = model_output
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": hyp,
        "ref": data.get("label", ""),
        "score": " ".join(str(score) for score in scores.tolist()),
    }
