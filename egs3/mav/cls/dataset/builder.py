"""MAV dataset builder."""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import subprocess
import zipfile
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `Sounds.csv` columns used: Filename (`6_anger.wav`), Actor_ID, Emotion.
_CSV_COLUMNS = ("Filename", "Actor_ID", "Emotion")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the MAV source tree, in order.

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
    """Return the corpus entries that are absent or incomplete under ``root``."""
    missing = []
    table = root / str(_CFG["table_name"])
    if not table.is_file():
        missing.append(str(table))
    sound_dir = root / str(_CFG["audio_subdir"])
    n_wav = sum(1 for _ in sound_dir.glob("*.wav")) if sound_dir.is_dir() else 0
    if n_wav < int(_CFG["expected_files"]):
        missing.append(f"{sound_dir} ({n_wav}/{_CFG['expected_files']} wav files)")
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the table and every recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding ``Sounds.csv`` and ``Sounds/``.

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
        "Complete MAV corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['table_name']}` and `{_CFG['audio_subdir']}/`, or let "
        "`prepare_source` download them."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$MAV_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_wavs(archive: Path, destination: Path) -> None:
    """Extract the WAV files of the archive into ``destination``.

    Args:
        archive: ``.zip`` archive to read.
        destination: Directory to extract into.

    Raises:
        zipfile.BadZipFile: If the archive is invalid.
        ValueError: If a member would be written outside ``destination``.
    """
    logger.info("Extracting %s", archive.name)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".wav"):
                continue
            if member.startswith("__MACOSX/"):
                continue
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member}")
            zf.extract(member, destination)


def _fetch_checked(name: str, md5: str, destination: Path) -> Path:
    """Download one Zenodo file and verify its md5; return the local path."""
    url = str(_CFG["base_url"]).rstrip("/") + "/" + name + "/content"
    path = destination / name
    part = path.with_suffix(path.suffix + ".part")
    download_url(url, part, logger=logger)
    digest = hashlib.md5(part.read_bytes()).hexdigest()
    if digest != md5:
        raise ValueError(f"MD5 mismatch for {name}: {digest}")
    part.rename(path)
    return path


def _download_source(destination: Path) -> None:
    """Fetch the label table and the recordings archive, then unpack the latter.

    Args:
        destination: Directory to fill.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a checksum does not match.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    _fetch_checked(str(_CFG["table_name"]), str(_CFG["table_md5"]), destination)
    archive = _fetch_checked(
        str(_CFG["archive_name"]), str(_CFG["archive_md5"]), destination
    )
    _extract_wavs(archive, destination)
    if _CFG.get("remove_archive", False):
        archive.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read ``Sounds.csv`` into one row per recording.

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a row carries an unknown emotion, a column is missing,
            or two rows share a file name.
        FileNotFoundError: If a recording named in the table is absent.
    """
    sound_dir = source_root / str(_CFG["audio_subdir"])
    entries = []
    seen: set[str] = set()
    with (source_root / str(_CFG["table_name"])).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in _CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Columns missing from the table: {missing}")
        for row in reader:
            name = row["Filename"].strip()
            emotion = row["Emotion"].strip()
            if emotion not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {emotion!r} for {name}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            if name in seen:
                raise ValueError(f"Duplicate recording name: {name}")
            seen.add(name)
            wav = sound_dir / name
            if not wav.is_file():
                raise FileNotFoundError(
                    f"Recording named in the table is absent: {wav}"
                )
            entries.append(
                {
                    "utt_id": f"mav-{Path(name).stem}",
                    "source": str(wav),
                    "speaker": row["Actor_ID"].strip(),
                    "label": _EMOTION_MAP[emotion],
                }
            )
    logger.info(
        "Read %d utterance(s) from %d actor(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole actors: no actor appears in two splits.

    Actors are shuffled with ``split_seed`` and cut by ``valid_ratio`` and
    ``test_ratio`` of the actor count. Every actor recorded each emotion once,
    so every emotion appears in every split.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If the three ratios do not sum to 1.
    """
    train_ratio = float(_CFG["train_ratio"])
    valid_ratio = float(_CFG["valid_ratio"])
    test_ratio = float(_CFG["test_ratio"])
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            "`train_ratio`/`valid_ratio`/`test_ratio` must sum to 1, got "
            f"{train_ratio} + {valid_ratio} + {test_ratio}"
        )

    speakers = sorted({e["speaker"] for e in entries}, key=int)
    Random(int(_CFG["split_seed"])).shuffle(speakers)
    n_test = round(len(speakers) * test_ratio)
    n_valid = round(len(speakers) * valid_ratio)
    split_of = {}
    for i, speaker in enumerate(speakers):
        if i < n_test:
            split_of[speaker] = "test"
        elif i < n_test + n_valid:
            split_of[speaker] = "valid"
        else:
            split_of[speaker] = "train"
    return {e["utt_id"]: split_of[e["speaker"]] for e in entries}


class MavBuilder(DatasetBuilder):
    """Prepare MAV for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one manifest
    per split, holding out whole actors (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a MAV source tree is reachable."""
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
            logger.info("MAV already present; skipping download")
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

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            wav = wav_dir / f"{entry['utt_id']}.wav"
            if not wav.is_file():
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
