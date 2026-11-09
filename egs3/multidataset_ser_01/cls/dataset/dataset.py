"""Dataset of experiment multidataset_ser_01 (not defined yet)."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset as TorchDataset


class MultidatasetSer01Dataset(TorchDataset):
    """Placeholder until the merged manifests exist.

    Which corpus is used for training and which for evaluation, and how their
    labels are unified, are still being decided (``docs/discussions/0003`` and
    ``0004`` of ser_dataset_db), so this recipe can only run ``create_dataset``.

    Raises:
        NotImplementedError: Always.
    """

    def __init__(self, split: str, recipe_dir: str | Path | None = None) -> None:
        raise NotImplementedError(
            "multidataset_ser_01 has no training or evaluation manifests yet; "
            "only the create_dataset stage can run."
        )
