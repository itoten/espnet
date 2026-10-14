"""Multi-corpus dataset builder that uses every corpus in full (ood20).

The training corpora are merged, with every split each corpus recipe
defines, into one training manifest. Each evaluation set is written as its
own manifest, built from the source splits listed for it in
``dataset/config.yaml``. Rows are read from each corpus recipe's unfiltered
``manifest/`` directly, because that recipe's ``Dataset`` returns the
length-filtered copy for ``train`` and ``valid``.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from importlib import import_module, resources
from pathlib import Path
from typing import Sequence

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.components.data.dataset_module import load_dataset_module
from espnet3.components.data.multicorpus_manifest import write_manifest
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_LABEL_MAP = {str(k): str(v) for k, v in _CFG["label_map"].items()}


@dataclass(frozen=True)
class CorpusSplits:
    """Which splits of one corpus recipe feed one output manifest.

    Attributes:
        name: Short corpus identifier, used for logging and as the test set
            key of an evaluation set.
        data_src: Corpus recipe tag such as ``tess/cls``.
        splits: Split names from that recipe's ``builder.manifest_paths``.
    """

    name: str
    data_src: str
    splits: tuple[str, ...]


def build_train_sources() -> list[CorpusSplits]:
    """Return every training corpus with all of its splits.

    Returns:
        list[CorpusSplits]: One entry per ``train_corpora`` item.
    """
    return [
        CorpusSplits(str(c["name"]), str(c["data_src"]), ("train", "valid", "test"))
        for c in _CFG["train_corpora"]
    ]


def build_eval_sources() -> list[CorpusSplits]:
    """Return every evaluation set, keyed by its ``name``.

    Returns:
        list[CorpusSplits]: One entry per ``eval_sets`` item.
    """
    return [
        CorpusSplits(
            str(c["name"]), str(c["data_src"]), tuple(str(s) for s in c["splits"])
        )
        for c in _CFG["eval_sets"]
    ]


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
    paths = [str(p) for p in _CFG["manifest_paths"].values()]
    paths += [f"manifest/{source.name}.tsv" for source in build_eval_sources()]
    return paths


def load_raw_rows(data_src: str, split: str) -> list[tuple[str, str, str]]:
    """Read one split of a corpus recipe's unfiltered manifest.

    The corpus recipe's own ``DatasetBuilder`` runs first if its outputs are
    missing, exactly as using that corpus on its own would.

    Args:
        data_src: Corpus recipe tag, e.g. ``tess/cls``.
        split: Key of that recipe's ``builder.manifest_paths``.

    Returns:
        list[tuple[str, str, str]]: ``(utt_id, wav_path, label)`` rows.

    Raises:
        KeyError: If the corpus recipe defines no manifest for ``split``.
    """
    module = load_dataset_module(data_src=data_src)
    recipe_root = Path(module.__file__).resolve().parents[1]
    builder = module.DatasetBuilder()
    if not builder.is_source_prepared(recipe_dir=recipe_root):
        builder.prepare_source(recipe_dir=recipe_root)
    if not builder.is_built(recipe_dir=recipe_root):
        builder.build(recipe_dir=recipe_root)

    data_root = import_module(f"{module.__name__}.builder").resolve_data_root(
        recipe_root
    )
    corpus_cfg = load_config_with_defaults(
        str(recipe_root / "dataset" / "config.yaml"), resolve=False
    )["builder"]
    manifest_path = data_root / str(corpus_cfg["manifest_paths"][split])

    rows = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                utt_id, wav_path, label = line.split("\t", maxsplit=2)
                rows.append((utt_id, wav_path, label))
    return rows


def merge_rows(
    sources: Sequence[CorpusSplits], context: str, keep_unmapped: bool = False
) -> list[str]:
    """Merge the listed splits of several corpora into manifest lines.

    Labels are mapped through ``label_map``. A label that is not a key of it
    is dropped, or kept unchanged when ``keep_unmapped`` is set. Rows are
    sorted by utterance id.

    Args:
        sources: Corpora and splits to merge.
        context: Output name, used only in log messages.
        keep_unmapped: Keep rows whose label is not in ``label_map``.

    Returns:
        list[str]: Newline-terminated ``utt_id<TAB>wav_path<TAB>label`` lines.

    Raises:
        RuntimeError: If an utterance id appears twice.
        FileNotFoundError: If a listed WAV file does not exist.
    """
    rows: list[tuple[str, str, str]] = []
    seen: dict[str, str] = {}
    kept: Counter = Counter()
    dropped: Counter = Counter()
    missing: list[str] = []

    for source in sources:
        for split in source.splits:
            for utt_id, wav_path, label in load_raw_rows(source.data_src, split):
                mapped = _LABEL_MAP.get(label, label if keep_unmapped else None)
                if mapped is None:
                    dropped[(source.name, label)] += 1
                    continue
                if utt_id in seen:
                    raise RuntimeError(
                        f"Duplicate utterance id {utt_id!r} in {source.name!r} "
                        f"(also in {seen[utt_id]!r}) while building {context!r}."
                    )
                if not Path(wav_path).is_file():
                    missing.append(wav_path)
                seen[utt_id] = f"{source.name}/{split}"
                rows.append((utt_id, wav_path, mapped))
                kept[(source.name, mapped)] += 1

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} WAV file(s) listed for {context!r} do not exist, "
            f"e.g. {missing[:3]}"
        )

    for (name, label), n in sorted(dropped.items()):
        logger.info(
            "%s: %s: dropped %d '%s' (not in label_map)", context, name, n, label
        )
    for (name, label), n in sorted(kept.items()):
        logger.info("%s: %s: kept %d -> '%s'", context, name, n, label)

    rows.sort(key=lambda row: row[0])
    return [f"{utt}\t{wav}\t{label}\n" for utt, wav, label in rows]


class MulticorpusBuilder(DatasetBuilder):
    """Builds the full-size training manifest and one manifest per eval set.

    Source preparation is delegated to each corpus's own recipe inside
    :func:`load_raw_rows`, so this builder's own ``is_source_prepared`` /
    ``prepare_source`` are no-ops.
    """

    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Report always ready; each corpus prepares itself in `build()`."""
        return True

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        """No-op: source preparation happens per corpus inside `build()`."""
        return None

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether every manifest this builder writes already exists."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        return all((data_root / p).is_file() for p in _all_manifest_paths())

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Write the training, placeholder validation and eval-set manifests.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
        """
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        manifest_paths = _CFG["manifest_paths"]

        train_lines = merge_rows(build_train_sources(), "train")
        write_manifest(train_lines, data_root / str(manifest_paths["train"]))
        valid_lines = train_lines[: int(_CFG["placeholder_valid_size"])]
        write_manifest(valid_lines, data_root / str(manifest_paths["valid"]))

        for source in build_eval_sources():
            lines = merge_rows([source], source.name, keep_unmapped=True)
            write_manifest(lines, data_root / "manifest" / f"{source.name}.tsv")
