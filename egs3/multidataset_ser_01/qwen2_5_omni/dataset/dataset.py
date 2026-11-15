"""Evaluation corpora of multidataset_ser_01, each with its closed label set."""

from __future__ import annotations

from typing import Any

import numpy as np

from egs3.multidataset_ser_01.cls.dataset.builder import LABELS
from egs3.multidataset_ser_01.cls.dataset.dataset import MultidatasetSer01Dataset
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field

__all__ = ["VoxEmoEvalDataset"]


class VoxEmoEvalDataset(BaseDataset):
    """One evaluation corpus as the `cls` recipe prepared it, for prompting.

    The rows are those of ``MultidatasetSer01Dataset(split)`` (the `cls`
    recipe's filtered manifest of that corpus), so the evaluation sets are
    the same as for the trained models. Each item adds:

    - ``utt_id``: the manifest's utterance id.
    - ``label_set``: the corpus's closed label set, as VoxEmo puts in the
      prompt: the recipe's labels (``labels`` in the `cls` recipe's
      ``dataset/config.yaml``) that occur in the corpus, in that order,
      joined by ``", "``. It is taken from every row of the corpus, before
      any subsampling.

    Args:
        split: Name of an evaluation corpus, e.g. ``ravdess``.
        max_utterances: Keep at most this many rows, drawn uniformly at
            random without replacement with ``seed`` and kept in manifest
            order; ``None`` keeps every row.
        seed: Seed of the draw.

    Raises:
        ValueError: If a row's label is not one of the recipe's labels.

    Examples:
        >>> ds = VoxEmoEvalDataset("legov2", max_utterances=100)  # doctest: +SKIP
        >>> ds[0]["label_set"]  # doctest: +SKIP
        'neutral, angry'
    """

    fields = (
        Field("utt_id", "text"),
        Field("speech", "audio"),
        Field("label", "text"),
        Field("label_set", "text"),
    )

    def __init__(
        self, split: str, max_utterances: int | None = None, seed: int = 0
    ) -> None:
        """Read the corpus's rows and choose the ones to keep."""
        self.corpus = MultidatasetSer01Dataset(split)
        labels = self.corpus.labels
        present = set(labels)
        unknown = sorted(present - set(LABELS))
        if unknown:
            raise ValueError(f"{split}: labels outside {LABELS}: {unknown}")
        self.label_set = [label for label in LABELS if label in present]
        self.utt_ids = self.corpus.utt_ids

        num_rows = len(labels)
        if max_utterances is None or max_utterances >= num_rows:
            self.indices = list(range(num_rows))
        else:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(num_rows, size=int(max_utterances), replace=False)
            self.indices = sorted(int(i) for i in chosen)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.indices[int(idx)]
        item = dict(self.corpus[row])
        item["utt_id"] = self.utt_ids[row]
        item["label_set"] = ", ".join(self.label_set)
        return item
