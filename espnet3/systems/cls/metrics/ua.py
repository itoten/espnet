"""Unweighted accuracy metric for classification."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.cls.metrics.scoring_utils import (
    check_unknown_reference,
    drop_unknown_rows,
    require_sklearn,
    resolve_classes,
    single_labels,
)


class UA(BaseMetric):
    """Compute unweighted accuracy (UA), the macro average of per-class recall.

    Every class counts equally regardless of how many examples it has; the
    emotion recognition literature also calls it unweighted average recall
    (UAR).

    Declare it in ``conf/metrics.yaml``:

    .. code-block:: yaml

        - metric:
            _target_: espnet3.systems.cls.metrics.ua.UA
            token_list: data/token_list  # optional; fixes the class set
          inputs:
            ref: ref
            hyp: hyp
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("UA", "number"),)

    def __init__(
        self,
        token_list: Optional[str] = None,
        unknown_reference: str = "error",
    ) -> None:
        """Initialize the UA metric.

        Args:
            token_list: Optional path to the recipe's token list. It fixes the
                class set so unseen labels are caught; without it the classes
                are taken from the references.
            unknown_reference: What to do with a reference label that is not
                in ``token_list``: ``"error"`` (default), ``"score"`` (count
                it as a class no prediction can match) or ``"skip"`` (leave
                those utterances out).
        """
        self.token_list = token_list
        self.unknown_reference = check_unknown_reference(unknown_reference)
        super().__init__()
        self.add_state("refs", [], "cat")
        self.add_state("hyps", [], "cat")

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Keep the labels of one batch; the score needs every utterance.

        Args:
            ref: Reference labels, one per utterance.
            hyp: Predicted labels, aligned with ``ref``.
        """
        self.refs.extend(ref)
        self.hyps.extend(hyp)

    def compute(self) -> Dict[str, float]:
        """Return the macro-averaged score of every utterance added so far.

        Returns:
            Dict[str, float]: ``{"UA": <percentage>}``, rounded to 2 places.

        Raises:
            RuntimeError: If ``scikit-learn`` is not installed.
            ValueError: If an utterance carries multiple labels, or if a
                reference label is missing from ``token_list`` while
                ``unknown_reference`` is ``"error"``.
        """
        sklearn_metrics = require_sklearn()
        references = single_labels(self.refs)
        hypotheses = single_labels(self.hyps)
        if self.token_list is not None:
            references, (hypotheses,) = drop_unknown_rows(
                references, [hypotheses], self.token_list, self.unknown_reference
            )
        classes = resolve_classes(references, self.token_list, self.unknown_reference)
        score = sklearn_metrics.recall_score(
            references,
            hypotheses,
            labels=classes,
            average="macro",
            zero_division=0,
        )
        return {"UA": round(float(score) * 100, 2)}
