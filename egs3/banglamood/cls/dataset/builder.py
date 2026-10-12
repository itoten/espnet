"""BanglaMood dataset builder."""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}
_ATTEMPTS = 3
_MEMBER_PATTERN = re.compile(r"(?:[^/]+/)*([A-Za-z]+)/(Female|Male)/(\d+\.wav)")


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the BanglaMood source tree, in order.

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
    """Return the first candidate holding all BanglaMood recordings.

    Recordings are looked up in the emotion folders below each candidate.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with ``listed_wavs`` recordings in its emotion
        folders.

    Raises:
        FileNotFoundError: If no candidate holds ``listed_wavs`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir() and len(_list_wavs(candidate)) >= _CFG["listed_wavs"]:
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete BanglaMood audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "BanglaMood recordings (one folder per emotion), or let `prepare_source` "
        "download them from Mendeley Data."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$BANGLAMOOD_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": str(_CFG["user_agent"])})


def _list_checksums() -> collections.Counter:
    """Look up the SHA-256 of every recording in the Mendeley Data listing.

    Returns:
        collections.Counter: SHA-256 digest to the number of listed files with
        that content.

    Raises:
        urllib.error.URLError: If the listing cannot be fetched.
    """
    with urllib.request.urlopen(_request(str(_CFG["dataset_url"])), timeout=60) as r:
        listing = json.load(r)["files"]
    return collections.Counter(f["content_details"]["sha256_hash"] for f in listing)


def _download_source(destination: Path) -> None:
    """Download the zip, check every recording against the listing, and unpack.

    The listing has no folder names, so each recording is matched by SHA-256 and
    the listing's digests must be used up exactly. Only members named
    ``<Emotion>/<Gender>/<digits>.wav`` are written, to paths built here, so the
    names inside the zip are not trusted. The zip is deleted afterwards.

    Args:
        destination: Directory that will hold one folder per emotion.

    Raises:
        OSError: If the download fails on every attempt.
        ValueError: If a recording is not in the listing, or the recordings do not
            add up to the listed ones.
    """
    destination.mkdir(parents=True, exist_ok=True)
    remaining = _list_checksums()
    archive = destination / "download.zip"
    part = archive.with_name(archive.name + ".part")
    logger.info("Downloading %s to %s", _CFG["archive_url"], destination)
    for attempt in range(_ATTEMPTS):
        try:
            with urllib.request.urlopen(
                _request(str(_CFG["archive_url"])), timeout=120
            ) as r:
                with part.open("wb") as dst:
                    shutil.copyfileobj(r, dst, 1 << 20)
            break
        except OSError:
            if attempt == _ATTEMPTS - 1:
                raise
            logger.warning("Download failed (attempt %d); retrying", attempt + 1)
            time.sleep(5 * (attempt + 1))
    part.replace(archive)

    written = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            match = _MEMBER_PATTERN.fullmatch(member)
            if match is None or match.group(1).lower() not in _EMOTION_MAP:
                continue
            data = zf.read(member)
            digest = hashlib.sha256(data).hexdigest()
            if remaining[digest] <= 0:
                raise ValueError(f"{member}: not in the Mendeley listing")
            remaining[digest] -= 1
            target = destination / match.group(1) / match.group(2) / match.group(3)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            written += 1
    if written != _CFG["listed_wavs"] or any(remaining.values()):
        raise ValueError(
            f"Extracted {written} recordings, expected {_CFG['listed_wavs']} "
            "matching the listing"
        )
    archive.unlink()


def _list_wavs(source_root: Path) -> list[Path]:
    """List the recordings, sorted by emotion folder, gender folder and name."""
    wavs = []
    for folder in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if folder.name.lower() in _EMOTION_MAP:
            wavs.extend(sorted(folder.glob("*/*.wav")))
    return wavs


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the distinct, unambiguous recordings, labeled by their folder.

    Recordings with identical content (SHA-256) are repeats. When they share a
    label, the first by folder and name is kept; when the labels differ, the
    recording is ambiguous and none of its copies is kept.

    Args:
        source_root: Corpus root holding one folder per emotion.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` and ``label``.

    Raises:
        FileNotFoundError: If no recording is found.
        ValueError: If the number of recordings left is not ``expected_files``.
    """
    groups: dict[str, list[tuple[str, Path]]] = {}
    for wav in _list_wavs(source_root):
        label = _EMOTION_MAP[wav.parent.parent.name.lower()]
        sha256 = hashlib.sha256(wav.read_bytes()).hexdigest()
        groups.setdefault(sha256, []).append((label, wav))
    if not groups:
        raise FileNotFoundError(f"No BanglaMood audio under {source_root}")

    entries = []
    for copies in groups.values():
        if len({label for label, _ in copies}) > 1:
            continue
        label, wav = copies[0]
        entries.append(
            {
                "utt_id": f"banglamood-{label}-{wav.stem}",
                "source": str(wav),
                "label": label,
            }
        )
    if len(entries) != _CFG["expected_files"]:
        raise ValueError(
            f"Found {len(entries)} usable recordings, expected "
            f"{_CFG['expected_files']}"
        )
    logger.info("Read %d utterance(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. The release does not say who
    spoke which recording, so speakers cannot be held out.

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

    rng = Random(int(_CFG["split_seed"]))
    by_label: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        by_label.setdefault(entry["label"], []).append(entry["utt_id"])

    assignment: dict[str, str] = {}
    for _, utt_ids in sorted(by_label.items()):
        shuffled = list(utt_ids)
        rng.shuffle(shuffled)
        n_valid = round(len(shuffled) * valid_ratio)
        n_test = round(len(shuffled) * test_ratio)
        for utt_id in shuffled[:n_test]:
            assignment[utt_id] = "test"
        for utt_id in shuffled[n_test : n_test + n_valid]:
            assignment[utt_id] = "valid"
        for utt_id in shuffled[n_test + n_valid :]:
            assignment[utt_id] = "train"
    return assignment


class BanglaMoodBuilder(DatasetBuilder):
    """Prepare BanglaMood for the classification recipe.

    `build()` converts every recording to 16 kHz mono with ffmpeg and writes one
    manifest per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether an BanglaMood source tree is reachable."""
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
            logger.info("BanglaMood already present; skipping download")
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
