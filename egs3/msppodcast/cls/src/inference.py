"""Output formatting for the emotion recipe's inference stage."""

from __future__ import annotations

from typing import Sequence

__all__ = ["build_output"]


def build_output(data: dict, model_output: Sequence, idx: int) -> dict:
    """Build one SCP record from a model output."""
    _, scores, hyp = model_output
    return {
        "utt_id": data.get("utt_id", str(idx)),
        "hyp": hyp,
        "ref": data.get("label", ""),
        "score": " ".join(str(score) for score in scores.tolist()),
    }
