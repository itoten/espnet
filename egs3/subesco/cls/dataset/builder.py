"""SUBESCO dataset builder."""

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

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `F_01_OISHI_S_1_ANGRY_1.wav` -> female speaker 1 (Oishi), sentence 1, angry,
# take 1. The trailing `]?` is not a typo here: one file in the corpus is named
# `F_02_MONIKA_S_2_SURPRISE_3].wav`.
FILENAME_RE = re.compile(
    r"^(?P<gender>[FM])_(?P<number>\d+)_(?P<name>[A-Z]+)_S_(?P<sentence>\d+)_"
    r"(?P<emotion>[A-Z]+)_(?P<take>\d+)\]?\.wav$"
)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the recordings, in order.

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


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding the WAV files.

    Raises:
        FileNotFoundError: If no candidate holds any.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir() and any(candidate.glob("*.wav")):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "SUBESCO recordings not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$SUBESCO_OUTPUT`` when set, else ``<recipe_dir>/data``.
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
    logger.info("Downloading %s (about 1.7 GB)", archive_name)
    download_url(url, part, logger=logger)
    part.rename(archive_path)
    _verify_md5(archive_path, str(_CFG["archive_md5"]))
    _extract_zip(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def parse_filename(name: str) -> dict[str, str] | None:
    """Split a SUBESCO file name into its fields.

    Args:
        name: File name such as ``F_01_OISHI_S_1_ANGRY_1.wav``.

    Returns:
        dict[str, str] | None: Keys ``gender``, ``number``, ``name``,
        ``sentence``, ``emotion`` and ``take``, or None when the name does not
        match the corpus scheme.

    Examples:
        >>> parse_filename("F_01_OISHI_S_1_ANGRY_1.wav")["emotion"]
        'ANGRY'
    """
    match = FILENAME_RE.match(name)
    return match.groupdict() if match else None


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking every field from the file name.

    Speaker numbers restart within each gender, so the speaker id joins the two:
    ``F_01`` and ``M_01`` are different people.

    Args:
        source_root: Directory holding the WAV files.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a name carries an emotion outside ``emotion_map``.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    skipped = 0
    for wav in sorted(source_root.glob("*.wav")):
        fields = parse_filename(wav.name)
        if fields is None:
            skipped += 1
            continue
        emotion = fields["emotion"]
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {emotion!r} in {wav.name}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        speaker = f"{fields['gender']}_{fields['number']}"
        entries.append(
            {
                "utt_id": (
                    f"subesco-{speaker}-S{fields['sentence']}"
                    f"-{emotion.lower()}-{fields['take']}"
                ),
                "source": str(wav),
                "speaker": speaker,
                "label": _EMOTION_MAP[emotion],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if not entries:
        raise FileNotFoundError(f"No SUBESCO audio under {source_root}")

    seen: dict[str, str] = {}
    for entry in entries:
        clash = seen.get(entry["utt_id"])
        if clash is not None:
            raise ValueError(
                f"Two recordings share the id {entry['utt_id']}: "
                f"{clash} and {entry['source']}"
            )
        seen[entry["utt_id"]] = entry["source"]
    logger.info(
        "Read %d recording(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(speakers: list[str]) -> dict[str, str]:
    """Deal speakers out to the splits, the same number of each gender.

    Args:
        speakers: Speaker id of every recording, repeats allowed.

    Returns:
        dict[str, str]: Speaker id to split name.

    Raises:
        ValueError: If the counts do not account for every speaker of a gender.
    """
    counts = {
        str(k): int(v) for k, v in _CFG["split_speaker_counts_per_gender"].items()
    }
    by_gender: dict[str, list[str]] = {}
    for speaker in sorted(set(speakers), key=lambda s: (s[0], int(s.split("_")[1]))):
        by_gender.setdefault(speaker[0], []).append(speaker)

    assignment = {}
    for gender, ordered in sorted(by_gender.items()):
        if sum(counts.values()) != len(ordered):
            raise ValueError(
                f"`split_speaker_counts_per_gender` sums to {sum(counts.values())} "
                f"but the corpus has {len(ordered)} speakers of gender {gender} "
                f"({', '.join(f'{k}={v}' for k, v in counts.items())}). "
                "Adjust the counts in dataset/config.yaml so every speaker "
                "lands in exactly one split."
            )
        start = 0
        for split, count in counts.items():
            for speaker in ordered[start : start + count]:
                assignment[speaker] = split
            start += count
    return assignment


class SUBESCOBuilder(DatasetBuilder):
    """Prepare SUBESCO for the classification recipe.

    The corpus ships 48 kHz 32-bit mono WAV, so `build()` resamples every
    recording to 16 kHz 16-bit and writes one manifest per split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a directory of recordings is reachable."""
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
            logger.info("SUBESCO already present; skipping download")
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
                        "-sample_fmt",
                        "s16",
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
