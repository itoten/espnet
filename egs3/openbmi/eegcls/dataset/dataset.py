"""OpenBMI dataset implementation backed by TSV manifests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from egs3.openbmi.eegcls.dataset.builder import resolve_data_root
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row: utterance id, epoch path, and class label."""

    utt_id: str
    epoch_path: Path
    label: str


def _read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Read ``utt_id<TAB>epoch_path<TAB>label`` lines as manifest entries."""
    entries: list[ManifestEntry] = []

    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            utt_id, epoch_path, label = line.split("\t", maxsplit=2)
            entries.append(ManifestEntry(utt_id, Path(epoch_path), label))

    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    return entries


class OpenBMIDataset(TorchDataset):
    """OpenBMI dataset that returns ``{"eeg", "label"}`` samples.

    The manifests must already exist: this class never builds them, so that
    each process of a multi-GPU run does not repeat the data preparation. Run
    the `create_dataset` stage first.

    Args:
        split: One of ``train``, ``valid``, or ``test``.
        recipe_dir: Optional recipe root. When omitted, defaults to the
            current recipe directory inferred from this module.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the manifest for ``split`` does not exist.

    Examples:
        >>> dataset = OpenBMIDataset(split="valid")
        >>> sorted(dataset[0].keys())
        ['eeg', 'label']
    """

    def __init__(self, split: str, recipe_dir: str | Path | None = None) -> None:
        self.split = str(split)
        if self.split not in _SPLIT_MANIFEST_PATHS:
            known = ", ".join(sorted(_SPLIT_MANIFEST_PATHS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = resolve_data_root(recipe_root)

        manifest_path = self.data_dir / _SPLIT_MANIFEST_PATHS[self.split]
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Run the `create_dataset` stage first."
            )

        self._entries = _read_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        return {
            "eeg": np.load(entry.epoch_path).astype(np.float32, copy=False),
            "label": entry.label,
        }
