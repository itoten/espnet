"""Output formatting for the VoxEmo inference of this recipe."""

from __future__ import annotations

import json
from typing import Any, Dict

__all__ = ["INVALID", "build_output"]

# Written as the hypothesis of a prompt whose answer could not be parsed.
INVALID = "<invalid>"


def build_output(data: Any, model_output: Any, idx: Any) -> Any:
    """Build the SCP records of one utterance, or of a batch.

    Fields, each written to ``<field>.scp``:

    - ``ref``: the reference label.
    - ``hyp``: the prompt ensemble's label.
    - ``hyp_<prompt>``: the label each prompt's answer names, or
      ``<invalid>`` when it cannot be parsed.
    - ``score``: the ensemble's distribution, as JSON ``{label: prob}``.
    - ``answer``: the raw answers, as JSON ``{prompt: text}`` on one line.

    Args:
        data: The dataset item, or a list of them for a batch.
        model_output: ``VoxEmoClassifier``'s result, or a list of them.
        idx: The dataset index, or a list of them.

    Returns:
        dict | list[dict]: One record, or one per utterance.
    """
    if isinstance(model_output, list):
        return [
            _build_record(item, output)
            for item, output in zip(data, model_output)
        ]
    return _build_record(data, model_output)


def _build_record(item: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, str]:
    """Return the fields of one utterance."""
    record = {
        "utt_id": item["utt_id"],
        "ref": item["label"],
        "hyp": output["prediction"],
        "score": json.dumps(
            dict(zip(output["labels"], map(float, output["distribution"])))
        ),
        "answer": json.dumps(output["answers"], ensure_ascii=False),
    }
    for variant, prediction in output["predictions"].items():
        record[f"hyp_{variant}"] = INVALID if prediction is None else prediction
    return record
