"""Talromur 3 dataset builder (Icelandic, six styles, speaker-held-out split)."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
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
_DROP_STYLES = {str(s) for s in _CFG["drop_styles"]}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the corpus, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate roots, each expected to hold ``recordings/``.
    """
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Root whose ``recordings/`` directory holds the speaker directories.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        recordings = candidate / str(_CFG["recordings_subdir"])
        n_flac = (
            sum(1 for _ in recordings.rglob("*.flac")) if recordings.is_dir() else 0
        )
        if n_flac >= int(_CFG["expected_files"]):
            return candidate
        checked.append(f"{candidate} ({n_flac}/{_CFG['expected_files']} flac files)")
    raise FileNotFoundError(
        "Complete Talromur 3 recordings not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the extracted "
        f"`{_CFG['recordings_subdir']}/`, or let `prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$TALROMUR3_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract(archive: Path, destination: Path) -> None:
    """Extract the FLAC files and index tables, after testing the archive.

    Args:
        archive: ``.zip`` archive to read.
        destination: Directory to extract into.

    Raises:
        zipfile.BadZipFile: If the archive is corrupt.
        ValueError: If a member would be written outside ``destination``.
    """
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"Corrupt member in {archive.name}: {bad}")
        for member in zf.namelist():
            if not member.endswith((".flac", "index.tsv")):
                continue
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member}")
            zf.extract(member, destination)


def _download_source(destination: Path) -> None:
    """Fetch the recordings archive and unpack it.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        zipfile.BadZipFile: If the archive is corrupt.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / str(_CFG["archive_name"])
    part = archive.with_suffix(archive.suffix + ".part")
    download_url(str(_CFG["archive_url"]), part, logger=logger)
    part.rename(archive)
    _extract(archive, destination)
    if _CFG.get("remove_archive", False):
        archive.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read every speaker's ``index.tsv`` into one row per emotional recording.

    Args:
        source_root: Root holding ``recordings/``.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a style is neither mapped nor dropped, or a file name
            is repeated.
        FileNotFoundError: If an indexed recording is absent or nothing is found.
    """
    recordings = source_root / str(_CFG["recordings_subdir"])
    entries = []
    seen: set[str] = set()
    dropped: dict[str, int] = {}
    for index in sorted(recordings.glob("*/index.tsv")):
        with index.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                name, speaker, style = row[0], row[1], row[2]
                if style in _DROP_STYLES:
                    dropped[style] = dropped.get(style, 0) + 1
                    continue
                if style not in _EMOTION_MAP:
                    raise ValueError(
                        f"Unknown style {style!r} in {index}. "
                        f"Known: {sorted(_EMOTION_MAP)}; "
                        f"dropped: {sorted(_DROP_STYLES)}"
                    )
                wav = index.parent / style / name
                if not wav.is_file():
                    raise FileNotFoundError(f"Indexed recording is absent: {wav}")
                if name in seen:
                    raise ValueError(f"Duplicate recording name: {name}")
                seen.add(name)
                entries.append(
                    {
                        "utt_id": f"talromur3-{Path(name).stem}",
                        "source": str(wav),
                        "speaker": speaker,
                        "label": _EMOTION_MAP[style],
                    }
                )
    if dropped:
        logger.info(
            "Dropped %d recording(s) that carry no emotion (%s)",
            sum(dropped.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())),
        )
    if not entries:
        raise FileNotFoundError(f"No Talromur 3 recordings under {recordings}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole speakers: ``test_speakers``, ``valid_speakers``, rest train.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If a listed speaker is absent or listed twice.
    """
    split_of = {str(s): "test" for s in _CFG["test_speakers"]}
    for speaker in _CFG["valid_speakers"]:
        if str(speaker) in split_of:
            raise ValueError(f"Speaker {speaker} is listed for two splits")
        split_of[str(speaker)] = "valid"
    present = {e["speaker"] for e in entries}
    absent = sorted(set(split_of) - present)
    if absent:
        raise ValueError(f"Listed speakers not in the corpus: {absent}")
    return {e["utt_id"]: split_of.get(e["speaker"], "train") for e in entries}


def _convert(ffmpeg: str, source: str, wav: Path, sampling_rate: str) -> None:
    """Convert one recording to mono WAV at ``sampling_rate`` unless it exists."""
    if wav.is_file():
        return
    part = wav.with_suffix(wav.suffix + ".part")
    subprocess.run(
        [
            ffmpeg,
            "-i",
            source,
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


class Talromur3Builder(DatasetBuilder):
    """Prepare Talromur 3 for the classification recipe.

    `build()` resamples every emotional recording to 16 kHz mono and writes one
    manifest per split, drawn as a speaker-held-out split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the recordings are reachable."""
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
            logger.info("Talromur 3 already present; skipping download")
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
        assignment = assign_splits(entries)

        with ThreadPoolExecutor(max_workers=int(_CFG["build_workers"])) as pool:
            list(
                pool.map(
                    lambda e: _convert(
                        ffmpeg,
                        e["source"],
                        wav_dir / f"{e['utt_id']}.wav",
                        sampling_rate,
                    ),
                    entries,
                )
            )

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            wav = wav_dir / f"{entry['utt_id']}.wav"
            rows[assignment[entry["utt_id"]]].append(
                (entry["utt_id"], str(wav.resolve()), entry["label"])
            )

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
