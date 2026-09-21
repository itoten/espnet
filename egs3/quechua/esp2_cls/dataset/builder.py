"""Quechua Collao dataset builder."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import zipfile
from importlib import resources
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

_XL = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTIONS = {str(e) for e in _CFG["emotions"]}
_ALIASES = {str(k): str(v) for k, v in _CFG["emotion_aliases"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the corpus tree, in order.

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
    missing = []
    if not (audio_dir.is_dir() and any(audio_dir.glob("*.wav"))):
        missing.append(str(audio_dir))
    metadata = root / str(_CFG["metadata_name"])
    if not metadata.is_file():
        missing.append(str(metadata))
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding both the audio and the table.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root containing ``Audios/`` and ``Data/``.

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
        "Quechua Collao source tree not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$QUECHUA_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _verify_md5(archive: Path, expected: str) -> None:
    """Check a downloaded archive against the checksum figshare publishes.

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
            f"{archive.name} has md5 {actual}, but the figshare record lists "
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
    archive_path = destination / str(_CFG["archive_name"])
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    logger.info("Downloading %s (about 3.8 GB)", archive_path.name)
    download_url(str(_CFG["download_url"]), part, logger=logger)
    part.rename(archive_path)
    _verify_md5(archive_path, str(_CFG["archive_md5"]))
    _extract_zip(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_sheet(path: Path, index: int) -> list[list[str | None]]:
    """Read one sheet of an xlsx workbook.

    An xlsx file is a zip of XML, so it is read here with the standard library
    rather than adding a spreadsheet dependency for one table.

    Args:
        path: Workbook to read.
        index: 1-based sheet number.

    Returns:
        list[list[str | None]]: Cell values, row by row.
    """
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in si.iter(f"{_XL}t"))
                for si in root.iter(f"{_XL}si")
            ]
        root = ET.fromstring(zf.read(f"xl/worksheets/sheet{index}.xml"))
    rows = []
    for row in root.iter(f"{_XL}row"):
        cells = []
        for cell in row.iter(f"{_XL}c"):
            value = cell.find(f"{_XL}v")
            text = value.text if value is not None else None
            if cell.get("t") == "s" and text is not None:
                text = shared[int(text)]
            cells.append(text)
        rows.append(cells)
    return rows


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the table into one row per recording, applying the repairs.

    Args:
        source_root: Corpus root.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a row carries an emotion outside ``emotions``.
        FileNotFoundError: If a recording named in the table is absent.
    """
    rows = read_sheet(
        source_root / str(_CFG["metadata_name"]), int(_CFG["metadata_sheet"])
    )
    fix_actor = {str(k): str(v) for k, v in _CFG["fix_actor"].items()}
    to_a5 = {str(x) for x in _CFG["reassign_to_a5"]}
    audio_dir = source_root / str(_CFG["audio_subdir"])

    entries = []
    n_fixed = n_moved = 0
    for row in rows[1:]:
        audio = str(int(float(row[0])))
        emotion = _ALIASES.get(row[1], row[1])
        if emotion not in _EMOTIONS:
            raise ValueError(
                f"Unknown emotion {row[1]!r} for recording {audio}. "
                f"Known: {sorted(_EMOTIONS)}"
            )
        speaker = row[2]
        if audio in to_a5:
            speaker, n_moved = "a5", n_moved + 1
        elif audio in fix_actor:
            speaker, n_fixed = fix_actor[audio], n_fixed + 1
        entries.append(
            {
                "utt_id": f"quechua-{speaker}-{emotion}-{audio}",
                "source": str(audio_dir / f"{audio}.wav"),
                "speaker": speaker,
                "label": emotion,
            }
        )
    if n_fixed:
        logger.info("Repaired the speaker id of %d row(s)", n_fixed)
    if n_moved:
        logger.info(
            "Reassigned %d recording(s) to a5; see dataset/config.yaml", n_moved
        )

    missing = [e for e in entries if not Path(e["source"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} recording(s) named in the table are absent under "
            f"{audio_dir}, starting with {Path(missing[0]['source']).name}."
        )
    logger.info(
        "Read %d recording(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(speakers: list[str]) -> dict[str, str]:
    """Deal sorted speaker ids out to the configured splits.

    Args:
        speakers: Speaker id of every recording, repeats allowed.

    Returns:
        dict[str, str]: Speaker id to split name.

    Raises:
        ValueError: If the counts do not account for every speaker.
    """
    counts = {str(k): int(v) for k, v in _CFG["split_speaker_counts"].items()}
    ordered = sorted(set(speakers))
    if sum(counts.values()) != len(ordered):
        raise ValueError(
            f"`split_speaker_counts` sums to {sum(counts.values())} but the "
            f"corpus has {len(ordered)} speakers ({', '.join(ordered)}). "
            "Adjust the counts in dataset/config.yaml so every speaker lands "
            "in exactly one split."
        )
    assignment = {}
    start = 0
    for split, count in counts.items():
        for speaker in ordered[start : start + count]:
            assignment[speaker] = split
        start += count
    return assignment


class QuechuaBuilder(DatasetBuilder):
    """Prepare the Quechua Collao corpus for the classification recipe.

    The recordings are mostly 44.1 kHz mono, with a few at other rates and in
    stereo, so `build()` puts every one through ffmpeg to 16 kHz mono.
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
            logger.info("Quechua Collao already present; skipping download")
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
        """Resample the recordings and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a resample fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required to resample the recordings to 16 kHz"
            )

        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)
        sampling_rate = str(_CFG["sampling_rate"])

        entries = read_entries(source_root)
        assignment = assign_splits([e["speaker"] for e in entries])

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            wav = wav_dir / f"{entry['utt_id']}.wav"
            if not wav.is_file():
                # `-f wav` sets the format, so the `.part` extension is fine.
                part = wav.with_suffix(wav.suffix + ".part")
                subprocess.run(
                    [
                        ffmpeg,
                        "-i",
                        entry["source"],
                        "-ac",
                        "1",
                        "-ar",
                        sampling_rate,
                        "-f",
                        "wav",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        str(part),
                    ],
                    check=True,
                )
                part.replace(wav)
            rows[assignment[entry["speaker"]]].append(
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
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
