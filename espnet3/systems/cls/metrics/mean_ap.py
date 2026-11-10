"""Mean average precision metric for classification."""

from __future__ import annotations

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


class MAP(BaseMetric):
    """Compute mean average precision (mAP) from class probabilities.

    Average precision is computed per class one-vs-rest and macro averaged
    over the classes the references contain, matching the ``mAP`` column of
    espnet2's ``cls_score.py``.

    It reads the per-class scores in addition to the reference labels.
    Declare it in ``conf/metrics.yaml``:

    .. code-block:: yaml

        - metric:
            _target_: espnet3.systems.cls.metrics.mean_ap.MAP
            token_list: data/token_list  # required; sets the column order
          inputs:
            ref: ref
            score: score
    """

    inputs = (Field("ref", "text"), Field("score", "text"))
    outputs = (Field("mAP", "number"),)

    def __init__(self, token_list: str, unknown_reference: str = "error") -> None:
        """Initialize the mAP metric.

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
            Dict[str, float]: ``{"mAP": <percentage>}``, rounded to 2 places.

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
        keep, _ = supported_classes(target, classes)
        values = [
            sklearn_metrics.average_precision_score(target[:, k], output[:, k])
            for k in keep
        ]
        return {"mAP": round(float(np.mean(values)) * 100, 2)}
