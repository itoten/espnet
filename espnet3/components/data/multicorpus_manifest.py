"""Merge label-mapped manifests from several recipe-local corpora into one.

A recipe that trains on several corpora at once (for example a cross-corpus
speech emotion recognition setup) cannot simply list each corpus under
``dataset.train`` in a training config: ``remove_long_short`` and
``prepare_labels`` each read one manifest per split
(``espnet3.systems.wavlmasp.system.WavlmAspSystem``), and dropping a row whose
label falls outside the target class set cannot be expressed as a per-sample
``transform``, which cannot change how many rows a dataset holds. This module
merges the corpora once, at ``DatasetBuilder.build()`` time, into a single
manifest per split, so the rest of the pipeline stays unmodified.

Every corpus recipe under ``egs3/`` built alongside this module shares one
``Dataset`` shape (see ``egs3/emodb/esp2_cls/dataset/dataset.py`` for the
reference implementation): its constructor takes ``split`` and exposes the
read manifest as ``_entries``, a list of objects carrying ``utt_id``,
``wav_path`` and ``label``. That attribute is private to each recipe, but the
shape is a deliberate, shared convention across every corpus recipe rather
than an accident of one module, which is why this function reaches into it
instead of re-deriving manifest paths on its own.

Most corpus recipes prefix ``utt_id`` with their own corpus name (e.g.
``emodb-03a02Lc``), but not all of them do (CREMA-D uses the bare WAV stem).
:func:`merge_train_valid_manifests` does not assume the prefix; it detects
an actual collision and raises instead.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from espnet3.components.data.dataset_module import load_dataset_module

logger = logging.getLogger(__name__)

__all__ = [
    "CorpusRef",
    "MergedSplit",
    "load_corpus_entries",
    "merge_train_valid_manifests",
    "build_eval_manifest",
    "write_manifest",
]


@dataclass(frozen=True)
class CorpusRef:
    """One corpus contributing to a merged manifest.

    Attributes:
        name: Short identifier used only for logging.
        data_src: Recipe/system tag such as ``cremad/esp2_cls``, resolved the
            same way ``dataset.train[*].data_src`` is (see
            ``espnet3.components.data.dataset_module``).
    """

    name: str
    data_src: str


@dataclass(frozen=True)
class MergedSplit:
    """One merged split: the manifest lines plus what happened to build them."""

    lines: list[str]
    kept: Counter
    dropped: Counter


def load_corpus_entries(data_src: str, split: str) -> list:
    """Return the manifest entries one corpus recipe holds for ``split``.

    Instantiating the recipe's ``Dataset`` triggers that corpus's own
    ``DatasetBuilder.prepare_source`` / ``build`` if its outputs are not
    already present, exactly as it would if the corpus were used on its own.

    Args:
        data_src: Recipe/system tag, e.g. ``cremad/esp2_cls``.
        split: Split name the corpus recipe recognises, e.g. ``train``.

    Returns:
        list: The corpus recipe's manifest entries (``utt_id``, ``wav_path``,
        ``label``).

    Raises:
        AttributeError: If the resolved recipe's ``Dataset`` does not expose
            ``_entries`` in the shared shape this module relies on.
    """
    module = load_dataset_module(data_src=data_src)
    dataset = module.Dataset(split=split)
    try:
        return dataset._entries
    except AttributeError as err:
        raise AttributeError(
            f"{data_src!r} Dataset(split={split!r}) has no `_entries`. "
            "multicorpus_manifest expects every corpus recipe's Dataset to "
            "follow the shared manifest-entry convention documented in this "
            "module."
        ) from err


def merge_train_valid_manifests(
    corpora: Sequence[CorpusRef],
    label_map: Mapping[str, str],
    splits: Sequence[str] = ("train", "valid"),
) -> dict[str, MergedSplit]:
    """Merge one split at a time across several corpora, applying ``label_map``.

    A row whose label is not a key of ``label_map`` is dropped. Utterance ids
    are expected to already be globally unique in practice (most corpus
    recipes under ``egs3/`` prefix them with their own corpus name; a
    collision most likely means the same corpus was listed twice), but this
    is checked rather than assumed.

    Args:
        corpora: Corpora to merge, in the order their rows appear if two
            utterance ids ever tie during the final sort (they do not, in
            practice, since ids are unique).
        label_map: Source label to target label. Labels outside this mapping
            are dropped rather than raising, since deciding which labels a
            multi-corpus class set keeps is a per-recipe judgement call.
        splits: Split names to read from each corpus and merge separately.

    Returns:
        dict[str, MergedSplit]: One entry per split, each holding the merged
        manifest lines (sorted by utterance id) and per-corpus kept/dropped
        label counts for logging.

    Raises:
        RuntimeError: If two corpora emit the same utterance id.
    """
    result: dict[str, MergedSplit] = {}
    for split in splits:
        rows: list[tuple[str, str, str]] = []
        seen: dict[str, str] = {}
        kept: Counter = Counter()
        dropped: Counter = Counter()

        for corpus in corpora:
            entries = load_corpus_entries(corpus.data_src, split)
            for entry in entries:
                mapped = label_map.get(entry.label)
                if mapped is None:
                    dropped[(corpus.name, entry.label)] += 1
                    continue
                if entry.utt_id in seen:
                    raise RuntimeError(
                        f"Duplicate utterance id {entry.utt_id!r} from "
                        f"{corpus.name!r} and {seen[entry.utt_id]!r} "
                        f"while merging split {split!r}."
                    )
                seen[entry.utt_id] = corpus.name
                rows.append((entry.utt_id, str(entry.wav_path), mapped))
                kept[(corpus.name, mapped)] += 1

        rows.sort(key=lambda row: row[0])
        lines = [f"{utt}\t{wav}\t{label}\n" for utt, wav, label in rows]
        result[split] = MergedSplit(lines=lines, kept=kept, dropped=dropped)

        for (name, label), n in sorted(dropped.items()):
            logger.info(
                "multicorpus[%s]: %s: dropped %d '%s' (not in label_map)",
                split,
                name,
                n,
                label,
            )
        for (name, label), n in sorted(kept.items()):
            logger.info("multicorpus[%s]: %s: kept %d -> '%s'", split, name, n, label)
        logger.info(
            "multicorpus[%s]: merged %d utterance(s) from %d corpus/corpora",
            split,
            len(lines),
            len(corpora),
        )

    return result


def build_eval_manifest(
    corpus: CorpusRef, split: str, label_map: Mapping[str, str]
) -> MergedSplit:
    """Build one label-mapped, unmerged manifest for a single corpus/split.

    Used for test sets, which stay separate per corpus rather than being
    combined: each ends up as its own named entry under ``dataset.test`` in
    the recipe's training config, so scores are reported per corpus.

    Args:
        corpus: Corpus to read.
        split: Split name the corpus recipe recognises, e.g. ``test``.
        label_map: Source label to target label; labels outside this mapping
            are dropped, same as :func:`merge_train_valid_manifests`.

    Returns:
        MergedSplit: The manifest lines (sorted by utterance id) plus
        kept/dropped label counts for logging.
    """
    merged = merge_train_valid_manifests([corpus], label_map, splits=(split,))
    return merged[split]


def write_manifest(lines: Sequence[str], manifest_path: Path) -> None:
    """Write manifest lines to ``manifest_path`` atomically.

    Writes to a ``.part`` file and renames it, so an interrupted write cannot
    leave a manifest that a later ``is_built()`` check would accept.

    Args:
        lines: Manifest lines, each already newline-terminated.
        manifest_path: Destination path.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    part = manifest_path.with_suffix(manifest_path.suffix + ".part")
    with part.open("w", encoding="utf-8") as fh:
        fh.writelines(lines)
    part.replace(manifest_path)
    logger.info("Wrote %d line(s) to %s", len(lines), manifest_path)
