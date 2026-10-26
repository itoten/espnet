"""MaSERC (Marathi) dataset builder."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# `student 91-Happy-1.wav`, `new 93 -fear-3.wav`, `new group 2 (soc1)-Sad-5.wav`:
# series and number identify the recording session, then the emotion word and a
# take number. A few names carry typos, a ` (2)` copy suffix or `Track 3`.
FILENAME_RE = re.compile(
    r"^(?P<prefix>.*?)\s*-\s*(?:Track \d+)?\s*(?P<emotion>[A-Za-z]+)\s*-\s*"
    r"(?P<take>\d+)(?:\s*\(\d+\))?\.wav$"
)
SESSION_RE = re.compile(r"^(?P<series>[A-Za-z]+(?: group)?)\s*(?P<number>\d+)")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the MaSERC source tree, in order.

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


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every MaSERC recording.

    Recordings are searched recursively, since they sit in a single
    `Entire Marathi Dataset Corpus/` directory.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with the ``.wav`` files somewhere below it.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and sum(1 for _ in candidate.rglob("*.wav")) >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete MaSERC audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the extracted "
        "`Entire Marathi Dataset Corpus/` directory, or let `prepare_source` "
        "download it from Mendeley Data."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$MASERC_OUTPUT`` when set, else ``<recipe_dir>/data``.
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


def _fetch(url: str, path: Path) -> None:
    """Stream ``url`` to ``path``.

    The host rejects urllib's default ``Python-urllib`` User-Agent with HTTP 403,
    so the request names itself instead.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "espnet3-recipe"})
    with urllib.request.urlopen(request) as response, path.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)


def _download_source(destination: Path) -> None:
    """Fetch the archive and unpack its WAV files.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = str(_CFG["archive_name"])
    url = str(_CFG["archive_url"])

    archive_path = destination / archive_name
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    _fetch(url, part)
    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    if digest != str(_CFG["archive_sha256"]):
        raise ValueError(f"SHA-256 mismatch for {archive_name}: {digest}")
    part.rename(archive_path)
    _extract_wavs(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def _parse_name(wav: Path) -> tuple[str, str, str]:
    """Return ``(group, emotion label, take)`` for a recording name."""
    fields = FILENAME_RE.match(wav.name)
    session = SESSION_RE.match(fields["prefix"].lower()) if fields else None
    if fields is None or session is None:
        raise ValueError(f"Unexpected MaSERC file name: {wav.name}")
    emotion = fields["emotion"].lower()
    if emotion not in _EMOTION_MAP:
        raise ValueError(
            f"Unknown emotion {emotion!r} in {wav.name}. Known: {sorted(_EMOTION_MAP)}"
        )
    series = "student" if session["series"].startswith("stu") else "new"
    if session["series"].endswith("group"):
        series = "newgroup"
    return f"{series}{session['number']}", _EMOTION_MAP[emotion], fields["take"]


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the usable recordings, taking the label from the file name.

    Empty files are skipped. Files with identical audio are collapsed to the
    first by name, or all dropped when their name labels disagree.

    Args:
        source_root: Corpus root with the recordings somewhere below it.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``group`` and
        ``label``. ``group`` is the session (series and number) in the name.

    Raises:
        ValueError: If a file name cannot be parsed or names an unknown emotion.
        FileNotFoundError: If no recording is found.
    """
    parsed = []
    empty = 0
    for wav in sorted(source_root.rglob("*.wav")):
        group, label, _take = _parse_name(wav)
        if sf.info(str(wav)).frames == 0:
            empty += 1
            continue
        digest = hashlib.sha256(wav.read_bytes()).hexdigest()
        parsed.append((wav, group, label, digest))

    by_digest: dict[str, list[tuple[Path, str, str, str]]] = {}
    for item in parsed:
        by_digest.setdefault(item[3], []).append(item)
    entries = []
    n_dup = n_conflict = 0
    counter: dict[tuple[str, str], int] = {}
    for wav, group, label, digest in parsed:
        same = by_digest[digest]
        if len({item[2] for item in same}) > 1:
            n_conflict += 1
            continue
        if _CFG["exclude_duplicates"] and same[0][0] != wav:
            n_dup += 1
            continue
        take = counter[(group, label)] = counter.get((group, label), 0) + 1
        entries.append(
            {
                "utt_id": f"maserc-{group}-{label}-{take}",
                "source": str(wav),
                "group": group,
                "label": label,
            }
        )
    if not entries:
        raise FileNotFoundError(f"No MaSERC audio under {source_root}")
    logger.info(
        "Read %d utterance(s) in %d session(s); skipped %d empty, %d duplicate "
        "and %d label-conflicting duplicate file(s)",
        len(entries),
        len({e["group"] for e in entries}),
        empty,
        n_dup,
        n_conflict,
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split whole sessions at random so no session spans two splits.

    Sessions are shuffled with ``split_seed`` and filled into test, then valid,
    then train until each holds about its ratio of the utterances. Labels are
    not balanced across splits.

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

    by_group: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        by_group.setdefault(entry["group"], []).append(entry["utt_id"])
    groups = sorted(by_group)
    Random(int(_CFG["split_seed"])).shuffle(groups)

    total = len(entries)
    quota = {"test": test_ratio * total, "valid": valid_ratio * total}
    counts = {"test": 0, "valid": 0}
    assignment: dict[str, str] = {}
    for group in groups:
        for split in ("test", "valid"):
            if counts[split] < quota[split]:
                break
        else:
            split = "train"
        if split != "train":
            counts[split] += len(by_group[group])
        for utt_id in by_group[group]:
            assignment[utt_id] = split
    return assignment


class MasercBuilder(DatasetBuilder):
    """Prepare MaSERC for the classification recipe.

    `build()` resamples every recording from 44.1 kHz stereo to 16 kHz mono and
    writes one manifest per split, drawn as a random split of whole sessions
    (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a MaSERC source tree is reachable."""
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
            logger.info("MaSERC already present; skipping download")
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
