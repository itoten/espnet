"""UrduSER dataset builder."""

from __future__ import annotations

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
_RECORDINGS_DIR = Path(str(_CFG["recordings_dir"]))
_ATTEMPTS = 4
_NAME_PATTERN = re.compile(r"(\d+)_([01])_([1-7])_(\d+)\.wav")


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the UrduSER source tree, in order.

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
    """Return the first candidate holding all UrduSER recordings.

    Recordings are looked up in ``recordings_dir`` below each candidate.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with at least ``expected_files`` recordings in
        ``recordings_dir``.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir() and len(_list_wavs(candidate)) >= _CFG["expected_files"]:
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete UrduSER audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "UrduSER recordings (`audio/*.wav`), or let `prepare_source` download "
        "them from Mendeley Data."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$URDUSER_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": str(_CFG["user_agent"])})


def _list_files() -> list[dict]:
    """List the recordings of the dataset.

    The listing is the Mendeley Data API's. Only files named like
    ``<actor>_<gender>_<emotion>_<number>.wav`` are taken, and the name is the one
    the file is stored under.

    Returns:
        list[dict]: Keys ``filename``, ``sha256``, ``size`` and ``url``.

    Raises:
        urllib.error.URLError: If the listing cannot be fetched.
        ValueError: If the number of recordings is not ``expected_files``
            (a different count would mean a changed release).
    """
    with urllib.request.urlopen(_request(str(_CFG["dataset_url"])), timeout=60) as r:
        listing = json.load(r)["files"]
    files = []
    for item in sorted(listing, key=lambda f: f["filename"]):
        if _NAME_PATTERN.fullmatch(item["filename"]) is None:
            continue
        details = item["content_details"]
        files.append(
            {
                "filename": item["filename"],
                "sha256": details["sha256_hash"],
                "size": details["size"],
                "url": details["download_url"],
            }
        )
    if len(files) != _CFG["expected_files"]:
        raise ValueError(
            f"Listed {len(files)} recordings, expected {_CFG['expected_files']}"
        )
    return files


def _download_source(destination: Path) -> None:
    """Download the dataset's zip, check each recording, and unpack them.

    Each recording is checked against the size and SHA-256 in the Mendeley Data
    listing. Only members named like
    ``<Emotion>/<actor>_<gender>_<emotion>_<number>.wav`` are written, to paths
    built here, so the names inside the zip are not trusted. The zip is deleted
    afterwards.

    Args:
        destination: Directory that will hold ``recordings_dir``.

    Raises:
        OSError: If the download fails on every attempt.
        ValueError: If a recording does not match its listing, or the zip does
            not hold every listed recording exactly once.
    """
    destination.mkdir(parents=True, exist_ok=True)
    listed = {entry["filename"]: entry for entry in _list_files()}
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

    target_dir = destination / _RECORDINGS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    written = set()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            name = member.rsplit("/", 1)[-1]
            if not member.endswith(".wav") or name not in listed:
                continue
            entry = listed[name]
            data = zf.read(member)
            if len(data) != entry["size"] or (
                hashlib.sha256(data).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"{name}: size or SHA-256 does not match")
            if name in written:
                raise ValueError(f"{name}: appears twice in the zip")
            (target_dir / name).write_bytes(data)
            written.add(name)
    if len(written) != len(listed):
        raise ValueError(f"Extracted {len(written)} recordings, expected {len(listed)}")
    archive.unlink()


def _list_wavs(source_root: Path) -> list[Path]:
    """List the recordings in ``recordings_dir``."""
    return sorted((source_root / _RECORDINGS_DIR).glob("*.wav"))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the emotion code in the name.

    Args:
        source_root: Corpus root holding ``recordings_dir``.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` and ``label``.

    Raises:
        ValueError: If a name is not ``<actor>_<gender>_<emotion>_<number>.wav``.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    for wav in _list_wavs(source_root):
        match = _NAME_PATTERN.fullmatch(wav.name)
        if match is None:
            raise ValueError(f"Unexpected file name: {wav}")
        entries.append(
            {
                "utt_id": f"urduser-{wav.stem}",
                "source": str(wav),
                "label": _EMOTION_MAP[match.group(3)],
            }
        )
    if not entries:
        raise FileNotFoundError(f"No UrduSER audio under {source_root}")
    logger.info("Read %d utterance(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. The actors named in the file
    names are not held out.

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


class UrduserBuilder(DatasetBuilder):
    """Prepare UrduSER for the classification recipe.

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
        """Check whether a UrduSER source tree is reachable."""
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
            logger.info("UrduSER already present; skipping download")
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
