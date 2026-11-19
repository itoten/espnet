"""VoxEmo's hard-label metrics for every prompt and the prompt ensemble."""

from __future__ import annotations

from typing import Dict, Sequence

from egs3.multidataset_ser_02.qwen2_5_omni.src.inference import INVALID
from egs3.multidataset_ser_02.qwen2_5_omni.src.voxemo import PROMPT_BLOCKS
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field

__all__ = ["VoxEmoScores"]

_SCORES = ("UA", "WA", "MicroF1", "MacroF1")


class VoxEmoScores(BaseMetric):
    """Score each VoxEmo prompt and the prompt ensemble on one test set.

    The hard-label metrics of VoxEmo (Section 3.4), in percent:

    - ``UA``: macro-averaged recall.
    - ``WA``: accuracy.
    - ``MicroF1`` and ``MacroF1``: F1 over the classes.
    - ``ParseFailure``: share of answers whose ``FINAL_LABEL`` could not be
      matched (per prompt only); they count as wrong in the other scores.

    The classes are those present in the references, as in the recipe's
    other metrics, so a class that a subsample happens to lack does not
    count. Every score is computed for each prompt (``<score>_<prompt>``,
    e.g. ``UA_t_a``) and for the ensemble (``<score>_ensemble``).

    One metric covers all of them because ``measure`` keys its results by
    metric class: the same class listed once per prompt would overwrite
    itself.

    Args:
        variants: The prompts that were run (keys of ``PROMPT_BLOCKS``).

    Examples:
        .. code-block:: yaml

            - metric:
                _target_: egs3.multidataset_ser_02.qwen2_5_omni.src.metrics.VoxEmoScores
              inputs:
                ref: ref
                hyp: hyp
                hyp_direct: hyp_direct
                ...
    """

    def __init__(self, variants: Sequence[str] = tuple(PROMPT_BLOCKS)) -> None:
        """Declare one input per prompt and the outputs that go with them."""
        unknown = sorted(set(variants) - set(PROMPT_BLOCKS))
        if unknown:
            raise ValueError(
                f"Unknown VoxEmo prompts {unknown}; choose from {list(PROMPT_BLOCKS)}"
            )
        self.variants = list(variants)
        self.inputs = (
            Field("ref", "text"),
            Field("hyp", "text"),
            *(Field(f"hyp_{v}", "text") for v in self.variants),
        )
        self.outputs = tuple(
            Field(f"{score}_{system}", "number")
            for system in [*self.variants, "ensemble"]
            for score in _SCORES
        ) + tuple(Field(f"ParseFailure_{v}", "number") for v in self.variants)
        super().__init__()
        for field in self.inputs:
            self.add_state(field.name, [], "cat")

    def update(self, **values: Sequence[str]) -> None:
        """Keep the labels of one chunk; the scores need every utterance."""
        for name, labels in values.items():
            getattr(self, name).extend(labels)

    def compute(self) -> Dict[str, float]:
        """Return every score, rounded to two decimals."""
        from sklearn import metrics as sk

        refs = list(self.ref)
        classes = sorted(set(refs))
        results = {}
        systems = {v: list(getattr(self, f"hyp_{v}")) for v in self.variants}
        systems["ensemble"] = list(self.hyp)
        for system, hyps in systems.items():
            kwargs = dict(labels=classes, zero_division=0)
            scores = {
                "UA": sk.recall_score(refs, hyps, average="macro", **kwargs),
                "WA": sk.accuracy_score(refs, hyps),
                "MicroF1": sk.f1_score(refs, hyps, average="micro", **kwargs),
                "MacroF1": sk.f1_score(refs, hyps, average="macro", **kwargs),
            }
            for score, value in scores.items():
                results[f"{score}_{system}"] = round(float(value) * 100, 2)
            if system != "ensemble":
                failures = sum(hyp == INVALID for hyp in hyps) / len(hyps)
                results[f"ParseFailure_{system}"] = round(failures * 100, 2)
        return results
