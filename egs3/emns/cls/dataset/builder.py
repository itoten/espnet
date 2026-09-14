"""EMNS dataset builder."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import tarfile
from importlib import resources
from pathlib import Path
from random import Random
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
    """Yield the directories that may hold the EMNS source tree, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate corpus roots, most specific first.
    """
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def missing_source_entries(root: Path) -> list[str]:
    """Return the corpus entries that are absent under ``root``.

    Args:
        root: Candidate corpus root.

    Returns:
        list[str]: Paths that a complete source tree would contain.
    """
    audio_dir = root / str(_CFG["audio_subdir"])
    metadata = root / str(_CFG["metadata_name"])
    missing = []
    if not (audio_dir.is_dir() and any(audio_dir.glob("*.webm"))):
        missing.append(str(audio_dir))
    if not metadata.is_file():
        missing.append(str(metadata))
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding both the audio and the metadata.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root containing ``cleaned_webm/`` and ``metadata.csv``.

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
        "EMNS source tree not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMNS_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_tar_xz(archive: Path, destination: Path) -> None:
    """Extract one xz-compressed tar archive into ``destination``.

    ``espnet3.utils.download_utils.extract_targz`` opens archives with
    ``"r:gz"`` and cannot read the ``.tar.xz`` OpenSLR publishes, so the
    extraction is done here. The ``filter="data"`` guard is kept: without it
    tarfile defaults to ``fully_trusted`` on Python < 3.14, and a downloaded
    archive could write outside ``destination``.

    Args:
        archive: Archive to read.
        destination: Directory to extract into.

    Raises:
        tarfile.TarError: If the archive is invalid or extraction fails.
    """
    logger.info("Extracting %s", archive.name)
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(path=destination, filter="data")


def _download_source(destination: Path) -> None:
    """Fetch the metadata file and the trimmed-audio archive.

    Args:
        destination: Corpus root to populate.

    Raises:
        urllib.error.URLError: If a download fails.
        tarfile.TarError: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    base_url = str(_CFG["base_url"]).rstrip("/")
    metadata_name = str(_CFG["metadata_name"])
    archive_name = str(_CFG["archive_name"])

    metadata_path = destination / metadata_name
    if not metadata_path.is_file():
        part = metadata_path.with_suffix(metadata_path.suffix + ".part")
        download_url(f"{base_url}/{metadata_name}", part, logger=logger)
        part.rename(metadata_path)

    archive_path = destination / archive_name
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    download_url(f"{base_url}/{archive_name}", part, logger=logger)
    part.rename(archive_path)
    _extract_tar_xz(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    """Read the rows that describe an existing recording.

    Rows whose ``status`` is not ``Complete`` have no audio: the file lists
    1205 rows for 1181 recordings, and the 24 extra rows are exactly those.

    Args:
        metadata_path: Path to ``metadata.csv``.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``file_name`` and ``label``.

    Raises:
        ValueError: If a kept row carries an emotion outside ``emotion_map``.
    """
    delimiter = str(_CFG["csv_delimiter"])
    status_column = str(_CFG["status_column"])
    status_keep = str(_CFG["status_keep"])
    audio_column = str(_CFG["audio_column"])
    emotion_column = str(_CFG["emotion_column"])
    id_column = str(_CFG["id_column"])

    entries = []
    n_incomplete = 0
    with metadata_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=delimiter):
            if row.get(status_column, "").strip() != status_keep:
                n_incomplete += 1
                continue
            emotion = row[emotion_column].strip()
            if emotion not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {emotion!r} in {metadata_path}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            # `audio_recording` says `wavs/...` but the files ship under
            # `cleaned_webm/`, so only the file name is meaningful.
            file_name = Path(row[audio_column].strip()).name
            entries.append(
                {
                    "utt_id": f"emns-{row[id_column].strip()}-{Path(file_name).stem}",
                    "file_name": file_name,
                    "label": _EMOTION_MAP[emotion],
                }
            )
    logger.info(
        "Read %d usable row(s); skipped %d without a recording",
        len(entries),
        n_incomplete,
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split utterances by emotion, keeping each split's label mix the same.

    EMNS holds one speaker, so a speaker-independent split cannot be built and
    the partition is stratified by emotion instead. **Scores from this recipe
    are therefore speaker-dependent** and are not comparable with a corpus
    whose test speakers are held out.

    Utterances are sorted by id before a seeded shuffle, so the assignment is a
    function of the corpus and ``split_seed`` alone.

    Args:
        entries: Rows from :func:`read_metadata`.

    Returns:
        dict[str, str]: Utterance id to split name.
    """
    ratios = {str(k): float(v) for k, v in _CFG["split_ratios"].items()}
    rng = Random(int(_CFG["split_seed"]))

    by_label: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        by_label.setdefault(entry["label"], []).append(entry["utt_id"])

    assignment = {}
    for label, utt_ids in sorted(by_label.items()):
        shuffled = list(utt_ids)
        rng.shuffle(shuffled)
        start = 0
        for split, ratio in ratios.items():
            count = int(len(shuffled) * ratio)
            for utt_id in shuffled[start : start + count]:
                assignment[utt_id] = split
            start += count
        for utt_id in shuffled[start:]:
            assignment[utt_id] = "train"
    return assignment


class EMNSBuilder(DatasetBuilder):
    """Prepare EMNS for the classification recipe.

    The corpus ships 48 kHz WebM, so `build()` transcodes every recording to
    16 kHz mono WAV and writes one manifest per split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete EMNS source tree is reachable."""
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
            logger.info("EMNS already present; skipping download")
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
        """Transcode the recordings and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a transcode fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required to convert the WebM recordings to WAV"
            )

        audio_dir = source_root / str(_CFG["audio_subdir"])
        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)
        sampling_rate = str(_CFG["sampling_rate"])

        entries = read_metadata(source_root / str(_CFG["metadata_name"]))
        # Drop rows without a recording before splitting, so a missing file
        # shrinks the corpus rather than skewing the split ratios.
        present = [e for e in entries if (audio_dir / e["file_name"]).is_file()]
        if len(present) < len(entries):
            logger.warning(
                "Skipped %d row(s) whose recording is missing",
                len(entries) - len(present),
            )
        entries = present
        assignment = assign_splits(entries)

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            source = audio_dir / entry["file_name"]
            wav = wav_dir / f"{entry['utt_id']}.wav"
            if not wav.is_file():
                # `-f wav` sets the format, so the `.part` extension is fine.
                part = wav.with_suffix(wav.suffix + ".part")
                subprocess.run(
                    [
                        ffmpeg,
                        "-i",
                        str(source),
                        "-ac",
                        "1",
                        "-ar",
                        sampling_rate,
                        "-f",
                        "wav",
                        "-vn",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        str(part),
                    ],
                    check=True,
                )
                part.replace(wav)
            rows[assignment[entry["utt_id"]]].append(
                (entry["utt_id"], str(wav.resolve()), entry["label"])
            )
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
            logger.info(
                "%s: wrote %d entries to %s",
                split,
                len(rows[split]),
                manifest,
            )
