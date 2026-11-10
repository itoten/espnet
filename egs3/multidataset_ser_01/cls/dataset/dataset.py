"""Dataset of experiment multidataset_ser_01 (not defined yet)."""

from __future__ import annotations

from pathlib import Path

from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field


class MultidatasetSer01Dataset(BaseDataset):
    """Placeholder until the merged manifests exist.

    Which corpus is used for training and which for evaluation, and how their
    labels are unified, are still being decided (``docs/discussions/0003`` and
    ``0004`` of ser_dataset_db), so this recipe can only run ``create_dataset``.

    Raises:
        NotImplementedError: Always.
    """

    fields = (Field("speech", "audio"), Field("label", "text"))

    def __init__(self, split: str, recipe_dir: str | Path | None = None) -> None:
        raise NotImplementedError(
            "multidataset_ser_01 has no training or evaluation manifests yet; "
            "only the create_dataset stage can run."
        )

    def __getitem__(self, index):
        """Not reached: ``__init__`` always raises."""
        raise NotImplementedError
