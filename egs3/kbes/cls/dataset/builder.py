"""KBES dataset builder."""

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
from concurrent.futures import ThreadPoolExecutor
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
_NAME_PATTERN = re.compile(r"([1-5])-([12])-([01])-(\d{3})\.wav")


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the KBES source tree, in order.

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
    """Return the first candidate holding all KBES recordings.

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
        "Complete KBES audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "KBES recordings (`audio/*.wav`), or let `prepare_source` "
        "download them from Mendeley Data."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$KBES_OUTPUT`` when set, else ``<recipe_dir>/data``.
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
    ``<emotion>-<intensity>-<gender>-<number>.wav`` are taken, and the name is the
    one the file is stored under.

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


def _fetch_file(entry: dict, destination: Path) -> None:
    """Download one recording and check its size and SHA-256.

    A failed transfer (for example a timeout) is retried a few times.

    Raises:
        ValueError: If the file does not match its listed size or SHA-256.
    """
    target = destination / _RECORDINGS_DIR / entry["filename"]
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(_ATTEMPTS):
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(_request(entry["url"]), timeout=120) as r:
                with part.open("wb") as dst:
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        digest.update(chunk)
                        dst.write(chunk)
            break
        except OSError:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(5 * (attempt + 1))
    if part.stat().st_size != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['filename']}: size or SHA-256 does not match")
    part.replace(target)


def _download_source(destination: Path) -> None:
    """Fetch every recording into ``destination``.

    Args:
        destination: Directory that will hold ``recordings_dir``.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a file does not match its listed size or SHA-256.
    """
    destination.mkdir(parents=True, exist_ok=True)
    files = _list_files()
    logger.info("Downloading %d recordings to %s", len(files), destination)
    with ThreadPoolExecutor(max_workers=int(_CFG["num_workers"])) as pool:
        list(pool.map(lambda entry: _fetch_file(entry, destination), files))


def _list_wavs(source_root: Path) -> list[Path]:
    """List the recordings in ``recordings_dir``."""
    return sorted((source_root / _RECORDINGS_DIR).glob("*.wav"))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the distinct, unambiguous recordings, labeled by their emotion code.

    Recordings with identical content (SHA-256) are repeats. When they share a
    label, the first by name is kept; when the labels differ, the recording is
    ambiguous and none of its copies is kept.

    Args:
        source_root: Corpus root holding ``recordings_dir``.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` and ``label``.

    Raises:
        ValueError: If a name is not ``<emotion>-<intensity>-<gender>-<number>.wav``
            or the number of recordings left is not ``expected_entries``.
        FileNotFoundError: If no recording is found.
    """
    groups: dict[str, list[tuple[str, Path]]] = {}
    for wav in _list_wavs(source_root):
        match = _NAME_PATTERN.fullmatch(wav.name)
        if match is None:
            raise ValueError(f"Unexpected file name: {wav}")
        sha256 = hashlib.sha256(wav.read_bytes()).hexdigest()
        groups.setdefault(sha256, []).append((_EMOTION_MAP[match.group(1)], wav))
    if not groups:
        raise FileNotFoundError(f"No KBES audio under {source_root}")

    entries = []
    for copies in groups.values():
        if len({label for label, _ in copies}) > 1:
            continue
        label, wav = copies[0]
        entries.append(
            {"utt_id": f"kbes-{wav.stem}", "source": str(wav), "label": label}
        )
    if len(entries) != _CFG["expected_entries"]:
        raise ValueError(
            f"Found {len(entries)} usable recordings, expected "
            f"{_CFG['expected_entries']}"
        )
    logger.info("Read %d utterance(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. The file names do not identify
    the actors, so none are held out.

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


class KBESBuilder(DatasetBuilder):
    """Prepare KBES for the classification recipe.

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
        """Check whether an KBES source tree is reachable."""
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
            logger.info("KBES already present; skipping download")
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
