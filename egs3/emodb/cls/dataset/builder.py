"""EmoDB dataset builder."""

from __future__ import annotations

import csv
import logging
import os
import zipfile
from importlib import resources
from pathlib import Path
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def _ambiguous_mode() -> str:
    """Return the configured handling of the ambiguous table.

    Raises:
        ValueError: If the mode is not one of ``none``, ``train`` or ``all``.
    """
    mode = str(_CFG["ambiguous"])
    if mode not in ("none", "train", "all"):
        raise ValueError(f"Unknown `ambiguous` {mode!r}; use 'none', 'train' or 'all'.")
    return mode


def _table_names() -> list[str]:
    """Return the metadata tables a complete source tree must contain."""
    names = [
        str(_CFG["files_table"]),
        str(_CFG["train_table"]),
        str(_CFG["test_table"]),
    ]
    if _ambiguous_mode() != "none":
        names.append(str(_CFG["ambiguous_table"]))
    return names


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the EmoDB source tree, in order.

    Each base is tried both directly and with the archive's own subdirectory
    appended, so an archive unpacked as-is and a flattened copy both resolve.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate corpus roots, most specific first.
    """
    bases = []
    if source_dir:
        bases.append(Path(source_dir).expanduser())
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        bases.append(Path(env_value).expanduser())
    bases.append(recipe_root / _CFG["dataset_path"])

    for base in bases:
        yield base
        yield base / str(_CFG["archive_subdir"])


def missing_source_entries(root: Path) -> list[str]:
    """Return the corpus entries that are absent under ``root``.

    Args:
        root: Candidate corpus root.

    Returns:
        list[str]: Paths that a complete source tree would contain.
    """
    audio_dir = root / str(_CFG["audio_subdir"])
    missing = []
    if not (audio_dir.is_dir() and any(audio_dir.glob("*.wav"))):
        missing.append(str(audio_dir))
    missing.extend(
        str(root / name) for name in _table_names() if not (root / name).is_file()
    )
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding both the audio and the tables.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root containing ``wav/`` and the ``db.*.csv`` tables.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        missing = missing_source_entries(candidate)
        if not missing:
            return candidate
        checked.append(f"{candidate} (missing: {', '.join(missing)})")
    raise FileNotFoundError(
        "EmoDB source tree not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMODB_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_zip(archive: Path, destination: Path) -> None:
    """Extract one zip archive into ``destination``.

    ``espnet3.utils.download_utils.extract_targz`` only opens gzipped tar
    archives, so the extraction is done here. Members are checked against the
    destination first: ``ZipFile.extractall`` does not reject absolute paths or
    ``..`` components on its own.

    Args:
        archive: Archive to read.
        destination: Directory to extract into.

    Raises:
        ValueError: If a member would be written outside ``destination``.
        zipfile.BadZipFile: If the archive is invalid.
    """
    logger.info("Extracting %s", archive.name)
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"{archive.name} escapes its destination: {name}")
        zf.extractall(path=destination)


def _download_source(destination: Path) -> None:
    """Fetch and unpack the corpus archive.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = str(_CFG["archive_name"])
    url = str(_CFG["base_url"]).rstrip("/") + "/" + archive_name

    archive_path = destination / archive_name
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    download_url(url, part, logger=logger)
    part.rename(archive_path)
    _extract_zip(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def _read_emotion_table(path: Path) -> dict[str, str]:
    """Map each file in one emotion table to its manifest label.

    Args:
        path: Table to read.

    Returns:
        dict[str, str]: Relative audio path to label.

    Raises:
        ValueError: If a row carries an emotion outside ``emotion_map``.
    """
    labels = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            emotion = row["emotion"].strip()
            if emotion not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {emotion!r} in {path}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            labels[row["file"].strip()] = _EMOTION_MAP[emotion]
    return labels


def read_tables(source_root: Path) -> list[dict[str, str]]:
    """Read the corpus tables into one row per utterance.

    The official split is taken from which emotion table a file appears in, so
    it stays tied to the corpus rather than to a list repeated in the recipe.
    Ambiguous samples, when enabled, are marked so the split can drop the ones
    belonging to a test speaker.

    Args:
        source_root: Corpus root holding ``wav/`` and the tables.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``rel_path``, ``speaker``,
        ``label`` and ``official``.
    """
    speakers = {}
    with (source_root / str(_CFG["files_table"])).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            speakers[row["file"].strip()] = row["speaker"].strip()

    tables = [
        ("train", _read_emotion_table(source_root / str(_CFG["train_table"]))),
        ("test", _read_emotion_table(source_root / str(_CFG["test_table"]))),
    ]
    if _ambiguous_mode() != "none":
        tables.append(
            (
                "ambiguous",
                _read_emotion_table(source_root / str(_CFG["ambiguous_table"])),
            )
        )

    entries = []
    for official, labels in tables:
        for rel_path, label in labels.items():
            entries.append(
                {
                    "utt_id": f"emodb-{Path(rel_path).stem}",
                    "rel_path": rel_path,
                    "speaker": speakers[rel_path],
                    "label": label,
                    "official": official,
                }
            )
    logger.info("Read %d utterance(s) from %d table(s)", len(entries), len(tables))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out the configured speakers from the official training split.

    The gold standard is passed through untouched, which keeps the reported
    scores comparable with work that uses the published partition. Ambiguous
    rows follow their speaker, so no split gains a speaker it did not have;
    under ``ambiguous: train`` the ones outside the training speakers are
    dropped instead.

    Args:
        entries: Rows from :func:`read_tables`.

    Returns:
        dict[str, str]: Utterance id to split name, omitting dropped rows.

    Raises:
        ValueError: If a configured validation speaker is not a train speaker.
    """
    mode = _ambiguous_mode()
    valid_speakers = {str(s) for s in _CFG["valid_speakers"]}
    train_speakers = {e["speaker"] for e in entries if e["official"] == "train"}
    unknown = valid_speakers - train_speakers
    if unknown:
        raise ValueError(
            f"`valid_speakers` names {sorted(unknown)}, which the corpus does "
            f"not place in the training split ({sorted(train_speakers, key=int)}). "
            "Holding out a test speaker would break the published partition."
        )

    assignment = {}
    dropped = 0
    for entry in entries:
        if entry["official"] == "test":
            assignment[entry["utt_id"]] = "test"
        elif entry["official"] == "ambiguous":
            if entry["speaker"] in valid_speakers:
                split = "valid"
            elif entry["speaker"] in train_speakers:
                split = "train"
            else:
                split = "test"
            if mode == "all" or split == "train":
                assignment[entry["utt_id"]] = split
            else:
                dropped += 1
        elif entry["speaker"] in valid_speakers:
            assignment[entry["utt_id"]] = "valid"
        else:
            assignment[entry["utt_id"]] = "train"
    if dropped:
        logger.info(
            "Dropped %d ambiguous utterance(s) outside the training speakers",
            dropped,
        )
    return assignment


class EmoDBBuilder(DatasetBuilder):
    """Prepare EmoDB for the classification recipe.

    The corpus ships 16 kHz mono WAV, so `build()` converts nothing and only
    writes one manifest per split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete EmoDB source tree is reachable."""
        try:
            resolve_source_root(Path(recipe_dir).resolve(), source_dir)
        except FileNotFoundError:
            return False
        return True

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Download and unpack the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("EmoDB already present; skipping download")
            return

        destination = next(iter_source_candidates(recipe_root, source_dir))
        _download_source(destination)
        resolve_source_root(recipe_root, source_dir)

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether every split manifest already exists."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        return all(
            (data_root / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus or a referenced recording is
                missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        entries = read_tables(source_root)
        missing = [e for e in entries if not (source_root / e["rel_path"]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} recording(s) named in the tables are absent "
                f"under {source_root}, starting with {missing[0]['rel_path']}. "
                "The archive is probably incomplete; remove it and let "
                "`prepare_source` fetch it again."
            )
        assignment = assign_splits(entries)

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            split = assignment.get(entry["utt_id"])
            if split is None:
                continue
            wav = (source_root / entry["rel_path"]).resolve()
            rows[split].append((entry["utt_id"], str(wav), entry["label"]))

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
