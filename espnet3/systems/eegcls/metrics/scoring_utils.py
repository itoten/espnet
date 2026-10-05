"""Shared helpers for classification metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

try:
    from sklearn import metrics as sklearn_metrics
except ImportError:
    sklearn_metrics = None

logger = logging.getLogger(__name__)

UNK_SYMBOL = "<unk>"


def require_sklearn():
    """Return ``sklearn.metrics``, raising a helpful error when unavailable.

    Returns:
        module: The ``sklearn.metrics`` module.

    Raises:
        RuntimeError: If ``scikit-learn`` is not installed.

    Example:
        >>> metrics = require_sklearn()
        >>> metrics.recall_score([0, 1], [0, 1], average="macro")
        1.0
    """
    if sklearn_metrics is None:
        raise RuntimeError(
            "scikit-learn is required for classification metrics. "
            "Please install it with `pip install scikit-learn`."
        )
    return sklearn_metrics


def load_class_labels(token_list: str | Path) -> List[str]:
    """Read a token list and return class labels in classifier output order.

    The final entry is dropped to mirror ``n_classes = len(token_list) - 1``
    in ``espnet2/tasks/cls.py``, where the trailing ``<unk>`` is not a class.
    A token list that does not end with ``<unk>`` is rejected, because
    dropping its last entry would silently discard a real class.

    Args:
        token_list: Path to the token list written by the ``prepare_labels``
            stage, which must be configured with ``add_symbol: ["<unk>:-1"]``.

    Returns:
        List[str]: Class labels ordered to match the score vector.

    Raises:
        ValueError: If the token list holds fewer than two entries, or does
            not end with ``<unk>``.

    Example:
        >>> load_class_labels("data/token_list")
        ['right', 'left']
    """
    path = Path(token_list)
    lines = path.read_text(encoding="utf-8").splitlines()
    tokens = [line.strip() for line in lines if line.strip()]
    if len(tokens) < 2:
        raise ValueError(
            f"token_list must hold at least two entries, got {len(tokens)}: {path}"
        )
    if tokens[-1] != UNK_SYMBOL:
        raise ValueError(
            f"token_list must end with {UNK_SYMBOL}, got {tokens[-1]!r}: {path}. "
            'Set add_symbol: ["<unk>:-1"] in the prepare_labels stage, '
            "otherwise the last label is dropped from the class list."
        )
    return tokens[:-1]


def single_labels(references: Sequence[str]) -> List[str]:
    """Validate that every SCP value holds exactly one label and return them.

    Args:
        references: One SCP value per utterance.

    Returns:
        List[str]: The single label of each utterance.

    Raises:
        ValueError: If any utterance carries zero or multiple labels.

    Example:
        >>> single_labels(["right", "left"])
        ['right', 'left']
        >>> single_labels(["right left"])
        Traceback (most recent call last):
        ValueError: Expected one label per utterance but row 0 has 2: ...
    """
    labels = []
    for row, reference in enumerate(references):
        parts = reference.split()
        if len(parts) != 1:
            raise ValueError(
                f"Expected one label per utterance but row {row} has "
                f"{len(parts)}: {reference!r}. This metric is multi-class only."
            )
        labels.append(parts[0])
    return labels


def resolve_classes(
    labels: Sequence[str], token_list: str | Path | None = None
) -> List[str]:
    """Return the classes to macro-average over, dropping unused ones.

    Args:
        labels: Reference label of each utterance.
        token_list: Optional token list fixing the class set and its order.
            When omitted, the classes are taken from ``labels`` in sorted
            order.

    Returns:
        List[str]: Classes that have at least one reference example.

    Raises:
        ValueError: If a reference label is absent from ``token_list``, or if
            ``labels`` is empty.

    Example:
        Without a token list the classes come from the references, sorted.

        >>> resolve_classes(["right", "left", "right"])
        ['left', 'right']
    """
    present = set(labels)
    if not present:
        raise ValueError("No reference labels to score")
    if token_list is not None:
        classes = load_class_labels(token_list)
        unknown = sorted(present - set(classes))
        if unknown:
            raise ValueError(
                f"Reference labels {unknown} are not in the token list: {classes}"
            )
    else:
        classes = sorted(present)
    dropped = [label for label in classes if label not in present]
    if dropped:
        logger.warning("Skipping classes with no reference example: %s", dropped)
    return [label for label in classes if label in present]
