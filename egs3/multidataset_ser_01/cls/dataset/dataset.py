"""Dataset of experiment multidataset_ser_01, backed by the merged manifests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from egs3.multidataset_ser_01.cls.dataset.builder import (
    EVAL_CORPORA,
    MultidatasetSer01Builder,
    eval_manifest_path,
    resolve_data_root,
)
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)

# `train`/`valid` read the `remove_long_short` output; each evaluation corpus
# reads its unfiltered manifest, under its own name.
_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _CONFIG["dataset"]["split_manifest_paths"].items()
}
_SPLIT_MANIFEST_PATHS.update({name: eval_manifest_path(name) for name in EVAL_CORPORA})


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row: utterance id, wav path, and emotion label."""

    utt_id: str
    wav_path: Path
    label: str


def _read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Read ``utt_id<TAB>wav_path<TAB>label`` lines as manifest entries."""
    entries: list[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path, label = line.split("\t", maxsplit=2)
            entries.append(ManifestEntry(utt_id, Path(wav_path), label))
    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return entries


class MultidatasetSer01Dataset(BaseDataset):
    """Merged dataset that returns ``{"speech", "label"}`` samples.

    Args:
        split: ``train``, ``valid``, or the name of an evaluation corpus
            (``eval_corpora`` in ``dataset/config.yaml``, e.g. ``ravdess``).
        recipe_dir: Optional recipe root. When omitted, defaults to the
            recipe directory inferred from this module.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the manifest for ``split`` does not exist.
    """

    fields = (Field("speech", "audio"), Field("label", "text"))

    def __init__(self, split: str, recipe_dir: str | Path | None = None) -> None:
        self.split = str(split)
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = resolve_data_root(recipe_root)

        if self.split not in _SPLIT_MANIFEST_PATHS:
            known = ", ".join(sorted(_SPLIT_MANIFEST_PATHS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        builder = MultidatasetSer01Builder()
        if not builder.is_built(recipe_dir=recipe_root):
            raise FileNotFoundError(
                f"The merged manifests under {self.data_dir} are missing or were "
                "built with other settings; run the create_dataset stage."
            )

        manifest_path = self.data_dir / _SPLIT_MANIFEST_PATHS[self.split]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self._entries = _read_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "label": entry.label,
        }
