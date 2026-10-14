"""MSP-Podcast dataset implementation backed by TSV manifests.

``train``/``valid`` are this recipe's own MSP-Podcast manifests. The 5
out-of-domain test splits (``emovdb``, ``iemocap``, ``tess``, ``savee``,
``ravdess``) instead read ``egs3/multicorpus_ood20/cls``'s already-built
manifests directly, the same way that recipe's own builder reads other
corpora's manifests (by locating the module and reading its files, not by
instantiating its ``Dataset``/``DatasetBuilder``).

This is deliberately *not* done by listing ``data_src:
multicorpus_ood20/cls`` on a ``dataset.test`` entry: the `create_dataset`
stage resolves every ``data_src`` it sees with *this* recipe's own
``recipe_dir`` (from `training_config.create_dataset.recipe_dir`), not the
referenced recipe's, so it ends up asking `multicorpus_ood20`'s
``MulticorpusBuilder`` to build into `this` recipe's ``data/`` directory
instead of reading `multicorpus_ood20`'s own. Reading the files directly here
sidesteps that stage entirely: nothing under ``multicorpus_ood20`` is ever
built, rebuilt or written to by this recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.msppodcast.cls.dataset.builder import (
    MSPPodcastBuilder,
    resolve_data_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}

# The 5 evaluation corpora, read from `multicorpus_ood20`'s own manifests
# (full-size, out-of-domain) rather than being rebuilt or copied here.
_OOD20_EVAL_SPLITS = {"emovdb", "iemocap", "tess", "savee", "ravdess"}


def _ood20_manifest_path(split: str) -> Path:
    """Return the path to one of `multicorpus_ood20`'s already-built manifests.

    Raises:
        FileNotFoundError: If that recipe has not been run yet.
    """
    module = import_module("egs3.multicorpus_ood20.cls.dataset.builder")
    ood20_recipe_root = Path(module.__file__).resolve().parents[1]
    path = module.resolve_data_root(ood20_recipe_root) / "manifest" / f"{split}.tsv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run `egs3/multicorpus_ood20/cls`'s "
            "create_dataset stage first; this recipe reads its manifests but "
            "never builds them."
        )
    return path


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
            entries.append(
                ManifestEntry(utt_id=utt_id, wav_path=Path(wav_path), label=label)
            )
    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return entries


class MSPPodcastDataset(TorchDataset):
    """MSP-Podcast (Train/Development, 8 classes) dataset.

    Args:
        split: ``train``, ``valid``, or one of the out-of-domain evaluation
            splits (``emovdb``, ``iemocap``, ``tess``, ``savee``, ``ravdess``).
        recipe_dir: Optional recipe root. When omitted, defaults to the
            current recipe directory inferred from this module.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the manifest for ``split`` does not exist.
    """

    def __init__(self, split: str, recipe_dir: str | Path | None = None) -> None:
        self.split = str(split)

        if self.split in _OOD20_EVAL_SPLITS:
            manifest_path = _ood20_manifest_path(self.split)
        else:
            recipe_root = (
                Path(recipe_dir).resolve()
                if recipe_dir is not None
                else Path(__file__).resolve().parents[1]
            )
            self.data_dir = resolve_data_root(recipe_root)

            builder = MSPPodcastBuilder()
            if not builder.is_source_prepared(recipe_dir=recipe_root):
                builder.prepare_source(recipe_dir=recipe_root)
            if not builder.is_built(recipe_dir=recipe_root):
                builder.build(recipe_dir=recipe_root)

            if self.split not in _SPLIT_MANIFEST_PATHS:
                known = ", ".join(
                    sorted(set(_SPLIT_MANIFEST_PATHS) | _OOD20_EVAL_SPLITS)
                )
                raise ValueError(
                    f"Unknown split '{self.split}'. Expected one of: {known}"
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
