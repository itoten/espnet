"""Weighted accuracy metric for classification."""

from __future__ import annotations

from typing import Dict, Sequence

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


class WA(BaseMetric):
    """Compute weighted accuracy (WA), the share of correct predictions.

    Every utterance counts once, so frequent classes dominate; read it next to
    :class:`espnet3.systems.cls.metrics.ua.UA`.

    Declare it in ``conf/metrics.yaml``:

    .. code-block:: yaml

        - metric:
            _target_: espnet3.systems.cls.metrics.wa.WA
          inputs:
            ref: ref
            hyp: hyp
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("WA", "number"),)

    def __init__(self) -> None:
        """Initialize the counters."""
        super().__init__()
        self.add_state("total", 0, "sum")
        self.add_state("correct", 0, "sum")

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Count the correct predictions of one batch.

        Args:
            ref: Reference labels, one per utterance.
            hyp: Predicted labels, aligned with ``ref``.

        Examples:
            >>> metric = WA()
            >>> metric.update(ref=["joy", "anger"], hyp=["joy", "joy"])
        """
        self.total += len(ref)
        self.correct += sum(int(r == h) for r, h in zip(ref, hyp))

    def compute(self) -> Dict[str, float]:
        """Return the accuracy over every utterance added so far.

        Returns:
            Dict[str, float]: ``{"WA": <percentage>}``, rounded to 2 places.

        Raises:
            ValueError: If no utterance was added.

        Examples:
            >>> metric = WA()
            >>> metric.update(ref=["joy", "anger"], hyp=["joy", "joy"])
            >>> metric.compute()
            {'WA': 50.0}
        """
        if self.total == 0:
            raise ValueError("No utterances to score")
        return {"WA": round(self.correct / self.total * 100, 2)}
