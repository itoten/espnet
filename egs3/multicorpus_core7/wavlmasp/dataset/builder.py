"""Multi-corpus dataset builder for the core7 label set.

Merges the training/validation manifests of every corpus in
``dataset/config.yaml``'s ``train_corpora`` into one manifest pair, applying
``label_map`` along the way, and writes one unmerged, label-mapped test
manifest per corpus/split. The merge logic itself lives in
``espnet3.components.data.multicorpus_manifest``; this module only wires that
shared logic to this recipe's own config.
"""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.components.data.multicorpus_manifest import (
    CorpusRef,
    build_eval_manifest,
    merge_train_valid_manifests,
    write_manifest,
)
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_LABEL_MAP = {str(k): str(v) for k, v in _CFG["label_map"].items()}


def _train_corpora() -> list[CorpusRef]:
    return [
        CorpusRef(str(c["name"]), str(c["data_src"])) for c in _CFG["train_corpora"]
    ]


def eval_manifest_specs() -> list[tuple[str, CorpusRef, str]]:
    """Return every (split_key, corpus, source_split) the test set needs.

    A corpus contributing a single eval split (every ``train_corpora`` entry,
    reading its own ``test`` split) is keyed by its own name, so
    ``dataset.test[*].data_src_args.split`` can just name the corpus. A
    corpus contributing several eval splits (MSP-Podcast's Test1/Test2) is
    keyed ``<corpus>_<split>`` instead, so the two stay distinguishable.

    Returns:
        list[tuple[str, CorpusRef, str]]: ``(split_key, corpus, source_split)``.
    """
    specs: list[tuple[str, CorpusRef, str]] = []
    for c in _CFG["train_corpora"]:
        corpus = CorpusRef(str(c["name"]), str(c["data_src"]))
        specs.append((corpus.name, corpus, "test"))
    for c in _CFG.get("eval_only_corpora", []):
        corpus = CorpusRef(str(c["name"]), str(c["data_src"]))
        splits = [str(s) for s in c["splits"]]
        for split in splits:
            key = corpus.name if len(splits) == 1 else f"{corpus.name}_{split}"
            specs.append((key, corpus, split))
    return specs


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the merged manifests are written into.

    Args:
        recipe_root: Recipe directory, used to resolve relative locations.

    Returns:
        Path: ``$<output_env_var>`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / str(_CFG["data_path"])


def _all_manifest_paths() -> list[str]:
    paths = list(str(p) for p in _CFG["manifest_paths"].values())
    paths += [f"manifest/{key}.tsv" for key, _corpus, _split in eval_manifest_specs()]
    return paths


class MulticorpusBuilder(DatasetBuilder):
    """Builds the merged train/valid manifest and the per-corpus test manifests.

    Source preparation is delegated entirely to each corpus's own recipe:
    instantiating that corpus's ``Dataset`` (done inside
    ``multicorpus_manifest``) triggers its own ``prepare_source`` / ``build``
    if needed, exactly as using that corpus on its own would. This builder's
    own ``is_source_prepared`` / ``prepare_source`` are therefore no-ops; all
    of the work happens in ``build()``.
    """

    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Report always ready; each sub-corpus decides for itself in `build()`."""
        return True

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        """No-op: source preparation happens per corpus inside `build()`."""
        return None

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether every merged/eval manifest already exists."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        return all((data_root / p).is_file() for p in _all_manifest_paths())

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Merge train/valid across corpora and write per-corpus test manifests.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.

        Raises:
            RuntimeError: If two corpora emit the same utterance id.
            AttributeError: If a referenced corpus recipe's `Dataset` does not
                follow the shared manifest-entry convention.
        """
        recipe_root = Path(recipe_dir).resolve()
        data_root = resolve_data_root(recipe_root)

        merged = merge_train_valid_manifests(
            _train_corpora(), _LABEL_MAP, splits=("train", "valid")
        )
        for split, relpath in _CFG["manifest_paths"].items():
            write_manifest(merged[str(split)].lines, data_root / str(relpath))

        for key, corpus, source_split in eval_manifest_specs():
            eval_manifest = build_eval_manifest(corpus, source_split, _LABEL_MAP)
            write_manifest(eval_manifest.lines, data_root / "manifest" / f"{key}.tsv")

        logger.info(
            "MulticorpusBuilder.build(): wrote merged train/valid and %d "
            "test manifest(s) under %s",
            len(eval_manifest_specs()),
            data_root,
        )
