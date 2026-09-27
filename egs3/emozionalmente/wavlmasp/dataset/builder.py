"""Emozionalmente dataset builder."""

from __future__ import annotations

import csv
import hashlib
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


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the corpus tree, in order.

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
        str(root / relpath)
        for relpath in _CFG["split_tables"].values()
        if not (root / str(relpath)).is_file()
    )
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding both the audio and the split tables.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root containing ``audio/`` and ``metadata/``.

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
        "Emozionalmente source tree not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMOZIONALMENTE_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _verify_md5(archive: Path, expected: str) -> None:
    """Check a downloaded archive against the checksum Zenodo publishes.

    Args:
        archive: Archive to read.
        expected: Hex digest from the record metadata.

    Raises:
        ValueError: If the digest does not match.
    """
    digest = hashlib.md5()
    with archive.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"{archive.name} has md5 {actual}, but the Zenodo record lists "
            f"{expected}. Delete it and let this stage fetch it again."
        )
    logger.info("Checksum verified: %s", actual)


def _extract_zip(archive: Path, destination: Path) -> None:
    """Extract one zip archive into ``destination``.

    Members are checked against the destination first: ``ZipFile.extractall``
    does not reject absolute paths or ``..`` components on its own.

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
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError(f"{archive.name} escapes its destination: {name}")
        zf.extractall(path=destination)


def _download_source(destination: Path) -> None:
    """Fetch, verify and unpack the corpus archive.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the checksum does not match.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = str(_CFG["archive_name"])
    url = str(_CFG["base_url"]).rstrip("/") + "/" + archive_name + "/content"

    archive_path = destination / archive_name
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    logger.info("Downloading %s (about 560 MB)", archive_name)
    download_url(url, part, logger=logger)
    part.rename(archive_path)
    _verify_md5(archive_path, str(_CFG["archive_md5"]))
    _extract_zip(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_split_table(path: Path) -> list[tuple[str, str]]:
    """Read one split table into ``(file name, label)`` pairs.

    Args:
        path: Table to read.

    Returns:
        list[tuple[str, str]]: One pair per row.

    Raises:
        ValueError: If a row carries an emotion outside ``emotion_map``.
    """
    file_column = str(_CFG["file_column"])
    emotion_column = str(_CFG["emotion_column"])
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            emotion = row[emotion_column].strip()
            if emotion not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {emotion!r} in {path}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            rows.append((row[file_column].strip(), _EMOTION_MAP[emotion]))
    return rows


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read every split table into one row per utterance.

    The split is taken from which table a recording appears in, so it stays
    tied to the corpus rather than to a list repeated in the recipe.

    Args:
        source_root: Corpus root holding ``audio/`` and ``metadata/``.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``label`` and
        ``split``.

    Raises:
        ValueError: If a recording appears in more than one split.
        FileNotFoundError: If a recording named in a table is absent.
    """
    audio_dir = source_root / str(_CFG["audio_subdir"])
    entries = []
    seen: dict[str, str] = {}
    for split, relpath in _CFG["split_tables"].items():
        for file_name, label in read_split_table(source_root / str(relpath)):
            clash = seen.get(file_name)
            if clash is not None:
                raise ValueError(
                    f"{file_name} is listed in both the {clash} and {split} "
                    "tables, which would put one recording in two splits."
                )
            seen[file_name] = split
            entries.append(
                {
                    "utt_id": f"emozionalmente-{Path(file_name).stem}",
                    "source": str(audio_dir / file_name),
                    "label": label,
                    "split": split,
                }
            )

    missing = [e for e in entries if not Path(e["source"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} recording(s) named in the split tables are absent "
            f"under {audio_dir}, starting with {Path(missing[0]['source']).name}. "
            "The archive is probably incomplete; remove it and let "
            "`prepare_source` fetch it again."
        )
    logger.info(
        "Read %d utterance(s) from %d table(s)",
        len(entries),
        len(_CFG["split_tables"]),
    )
    return entries


class EmozionalmenteBuilder(DatasetBuilder):
    """Prepare Emozionalmente for the classification recipe.

    The corpus ships 16 kHz mono WAV, so `build()` converts nothing and only
    writes one manifest per split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete source tree is reachable."""
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
            ValueError: If the archive fails its checksum.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("Emozionalmente already present; skipping download")
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

        entries = read_entries(source_root)
        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            wav = Path(entry["source"]).resolve()
            rows[entry["split"]].append((entry["utt_id"], str(wav), entry["label"]))

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
