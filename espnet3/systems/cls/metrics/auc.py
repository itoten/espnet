"""ROC AUC metric for classification."""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.cls.metrics.scoring_utils import (
    build_score_matrix,
    build_target_matrix,
    check_unknown_reference,
    drop_unknown_rows,
    load_class_labels,
    require_sklearn,
    supported_classes,
)

logger = logging.getLogger(__name__)


class AUC(BaseMetric):
    """Compute the macro-averaged one-vs-rest ROC AUC from class probabilities.

    Classes with no positive or no negative reference are skipped, since
    their AUC is undefined.

    It reads the per-class scores in addition to the reference labels.
    Declare it in ``conf/metrics.yaml``:

    .. code-block:: yaml

        - metric:
            _target_: espnet3.systems.cls.metrics.auc.AUC
            token_list: data/token_list  # required; sets the column order
          inputs:
            ref: ref
            score: score
    """

    inputs = (Field("ref", "text"), Field("score", "text"))
    outputs = (Field("AUC", "number"),)

    def __init__(self, token_list: str, unknown_reference: str = "error") -> None:
        """Initialize the AUC metric.

        Args:
            token_list: Path to the recipe's token list. It defines the order
                of the columns in ``score.scp``.
            unknown_reference: What to do with a reference label that is not
                in ``token_list``: ``"error"`` (default), ``"score"`` (treat
                the utterance as a negative for every class) or ``"skip"``
                (leave those utterances out).
        """
        self.token_list = token_list
        self.unknown_reference = check_unknown_reference(unknown_reference)
        super().__init__()
        self.add_state("refs", [], "cat")
        self.add_state("scores", [], "cat")

    def update(self, *, ref: Sequence[str], score: Sequence[str]) -> None:
        """Keep the labels and scores of one batch; the score needs them all.

        Args:
            ref: Reference labels, one per utterance.
            score: Space-separated class scores in ``token_list`` order.
        """
        self.refs.extend(ref)
        self.scores.extend(score)

    def compute(self) -> Dict[str, float]:
        """Return the macro average over the classes the references contain.

        Returns:
            Dict[str, float]: ``{"AUC": <percentage>}``, rounded to 2 places.

        Raises:
            RuntimeError: If ``scikit-learn`` is not installed.
            ValueError: If the scores and token list disagree, or a reference
                label is missing from ``token_list`` while
                ``unknown_reference`` is ``"error"``.
        """
        sklearn_metrics = require_sklearn()
        refs, (scores,) = drop_unknown_rows(
            list(self.refs),
            [list(self.scores)],
            self.token_list,
            self.unknown_reference,
        )
        classes = load_class_labels(self.token_list)
        target = build_target_matrix(
            refs, classes, allow_unknown=self.unknown_reference == "score"
        )
        output = build_score_matrix(scores, len(classes))
        keep, kept_labels = supported_classes(target, classes)
        scorable = [k for k in keep if target[:, k].sum() < target.shape[0]]
        if not scorable:
            raise ValueError(
                "AUC needs a class with both a positive and a negative "
                f"reference, but every utterance carries {kept_labels}"
            )
        dropped = [classes[k] for k in keep if k not in set(scorable)]
        if dropped:
            logger.warning("Skipping classes with no negative example: %s", dropped)
        values = [
            sklearn_metrics.roc_auc_score(target[:, k], output[:, k]) for k in scorable
        ]
        return {"AUC": round(float(np.mean(values)) * 100, 2)}
