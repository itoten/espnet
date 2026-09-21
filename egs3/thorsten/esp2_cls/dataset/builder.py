"""Thorsten-Emotional dataset builder."""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTIONS = [str(e) for e in _CFG["emotions"]]


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
    missing = [
        str(root / emotion)
        for emotion in _EMOTIONS
        if not (root / emotion).is_dir() or not any((root / emotion).glob("*.wav"))
    ]
    metadata = root / str(_CFG["metadata_name"])
    if not metadata.is_file():
        missing.append(str(metadata))
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every emotion folder and the table.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        missing = missing_source_entries(candidate)
        if not missing:
            return candidate
        checked.append(f"{candidate} (missing: {', '.join(missing[:3])})")
    raise FileNotFoundError(
        "Thorsten-Emotional source tree not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$THORSTEN_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _verify_md5(archive: Path, expected: str) -> None:
    """Check a downloaded archive against the published checksum.

    Args:
        archive: Archive to read.
        expected: Hex digest from the Zenodo record.

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


def _download_source(destination: Path) -> None:
    """Fetch, verify and unpack the corpus archive.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the checksum does not match.
        tarfile.TarError: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = str(_CFG["archive_name"])
    url = str(_CFG["base_url"]).rstrip("/") + "/" + archive_name

    archive_path = destination / archive_name
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    logger.info("Downloading %s (about 400 MB)", archive_name)
    download_url(url, part, logger=logger)
    part.rename(archive_path)
    _verify_md5(archive_path, str(_CFG["archive_md5"]))
    extract_targz(archive_path, destination, logger=logger)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_sentences(source_root: Path) -> dict[str, str]:
    """Read the hash-to-text table.

    Args:
        source_root: Corpus root.

    Returns:
        dict[str, str]: Sentence hash to its German text.
    """
    path = source_root / str(_CFG["metadata_name"])
    delimiter = str(_CFG["csv_delimiter"])
    sentences = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter=delimiter):
            if len(row) >= 2:
                sentences[row[0].strip()] = row[1].strip()
    logger.info("Read %d sentence(s) from %s", len(sentences), path.name)
    return sentences


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the directory name.

    Args:
        source_root: Corpus root.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``sentence`` and
        ``label``.

    Raises:
        FileNotFoundError: If an emotion folder holds no recording.
    """
    sentences = read_sentences(source_root)
    entries = []
    unknown = set()
    for emotion in _EMOTIONS:
        recordings = sorted((source_root / emotion).glob("*.wav"))
        if not recordings:
            raise FileNotFoundError(f"No recordings under {source_root / emotion}")
        for wav in recordings:
            if wav.stem not in sentences:
                unknown.add(wav.stem)
                continue
            entries.append(
                {
                    "utt_id": f"thorsten-{emotion}-{wav.stem}",
                    "source": str(wav),
                    "sentence": wav.stem,
                    "label": emotion,
                }
            )
    if unknown:
        logger.warning(
            "Skipped %d recording(s) whose sentence is not in %s",
            len(unknown),
            _CFG["metadata_name"],
        )

    # The corpus is documented as 2400 recordings, but `whisper` is one short.
    per_sentence = {}
    for entry in entries:
        per_sentence.setdefault(entry["sentence"], set()).add(entry["label"])
    short = {
        s: sorted(set(_EMOTIONS) - v)
        for s, v in per_sentence.items()
        if len(v) < len(_EMOTIONS)
    }
    for sentence, missing in short.items():
        logger.info("Sentence %s was not recorded in %s", sentence, ", ".join(missing))
    logger.info(
        "Read %d recording(s) over %d sentence(s)", len(entries), len(per_sentence)
    )
    return entries


def assign_splits(sentences: list[str]) -> dict[str, str]:
    """Split the sentences, keeping every emotion of one together.

    Args:
        sentences: Sentence hash of every recording, repeats allowed.

    Returns:
        dict[str, str]: Sentence hash to split name.

    Raises:
        ValueError: If the counts leave nothing to train on.
    """
    counts = {str(k): int(v) for k, v in _CFG["split_sentence_counts"].items()}
    ordered = sorted(set(sentences))
    if sum(counts.values()) >= len(ordered):
        raise ValueError(
            f"`split_sentence_counts` takes {sum(counts.values())} of the "
            f"{len(ordered)} sentences, leaving nothing for training. Lower "
            "the counts in dataset/config.yaml."
        )
    shuffled = list(ordered)
    Random(int(_CFG["split_seed"])).shuffle(shuffled)

    assignment = {}
    start = 0
    for split, count in counts.items():
        for sentence in shuffled[start : start + count]:
            assignment[sentence] = split
        start += count
    for sentence in shuffled[start:]:
        assignment[sentence] = "train"
    return assignment


class ThorstenBuilder(DatasetBuilder):
    """Prepare Thorsten-Emotional for the classification recipe.

    The corpus ships 22.05 kHz mono WAV, so `build()` resamples every recording
    to 16 kHz and writes one manifest per split.
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
            logger.info("Thorsten-Emotional already present; skipping download")
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
        assignment = assign_splits([e["sentence"] for e in entries])

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
            rows[assignment[entry["sentence"]]].append(
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
