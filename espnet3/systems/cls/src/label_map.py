"""Map a released classifier's labels onto a recipe's classes."""

from __future__ import annotations

import logging
from typing import List, Mapping, Optional, Sequence

import torch

__all__ = ["resolve_label_map", "gather_class_scores"]

logger = logging.getLogger(__name__)


def resolve_label_map(
    model_labels: Sequence[str],
    label_map: Mapping[str, str],
    classes: Sequence[str],
    model_name: str = "model",
) -> List[Optional[int]]:
    """Return, for each class, the index of the model label mapped to it.

    Used by zero-shot inference models that run a released classifier with
    its own label set and score it against a recipe's classes.

    Args:
        model_labels: The model's labels in its output order.
        label_map: Mapping from a model label to a class.
        classes: The recipe's classes in score order.
        model_name: Name used in the warning about unmapped classes.

    Returns:
        list: One entry per class: the model's output index, or ``None``
        when no model label maps to that class.

    Raises:
        ValueError: If ``label_map`` names an unknown label or class, or
            maps two labels to one class.

    Examples:
        >>> resolve_label_map(["Anger", "Other"], {"Anger": "angry"}, ["angry"])
        [0]
    """
    model_labels = list(model_labels)
    classes = list(classes)
    unknown_labels = sorted(set(label_map) - set(model_labels))
    if unknown_labels:
        raise ValueError(
            f"label_map names labels the model does not have: "
            f"{unknown_labels}; the model's labels are {model_labels}"
        )
    unknown_classes = sorted(set(label_map.values()) - set(classes))
    if unknown_classes:
        raise ValueError(
            f"label_map maps to classes not in token_list: "
            f"{unknown_classes}; the classes are {classes}"
        )
    source_index: List[Optional[int]] = [None] * len(classes)
    for label, cls in label_map.items():
        target = classes.index(cls)
        if source_index[target] is not None:
            raise ValueError(f"label_map maps more than one label to {cls!r}")
        source_index[target] = model_labels.index(label)

    unmapped = [c for c, i in zip(classes, source_index) if i is None]
    if unmapped:
        logger.warning(
            "No %s label maps to %s; these classes are never predicted.",
            model_name,
            unmapped,
        )
    return source_index


def gather_class_scores(
    probabilities: Sequence[float] | torch.Tensor,
    source_index: Sequence[Optional[int]],
) -> torch.Tensor:
    """Gather the mapped probabilities in class order and renormalise them.

    The probability of every model label that no class takes is discarded,
    and a class with no source label scores zero.

    Args:
        probabilities: The model's probabilities over its own labels.
        source_index: Output of :func:`resolve_label_map`.

    Returns:
        torch.Tensor: Per-class probabilities summing to one.

    Examples:
        >>> gather_class_scores([0.2, 0.6, 0.2], [0, 2])
        tensor([0.5000, 0.5000])
    """
    probabilities = [float(p) for p in probabilities]
    scores = torch.tensor(
        [0.0 if i is None else probabilities[i] for i in source_index],
        dtype=torch.float32,
    )
    return scores / scores.sum()
