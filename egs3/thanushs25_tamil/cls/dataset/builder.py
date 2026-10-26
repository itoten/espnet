"""Thanushs25 Tamil dataset builder (Hugging Face files, one folder per emotion)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import urllib.parse
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


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the source tree, in order.

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


def _list_wavs(root: Path) -> list[Path]:
    """Return the ``*.wav`` files below ``<root>/<audio_subdir>/<emotion>/``."""
    audio_dir = root / str(_CFG["audio_subdir"])
    return sorted(audio_dir.glob("*/*.wav")) if audio_dir.is_dir() else []


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding ``final_data/<emotion>/*.wav``.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        n_wav = len(_list_wavs(candidate))
        if n_wav >= int(_CFG["expected_files"]):
            return candidate
        checked.append(f"{candidate} ({n_wav}/{_CFG['expected_files']} wav files)")
    raise FileNotFoundError(
        "Complete corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['audio_subdir']}/<emotion>/*.wav`, or let `prepare_source` "
        "download them from the Hugging Face repository."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$THANUSHS25_TAMIL_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _fetch(repo: str, revision: str, item: dict, destination: Path) -> None:
    """Download one repository file and check its SHA-256 when the Hub lists one."""
    target = destination / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
        + urllib.parse.quote(item["path"])
    )
    part = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, part)
    expected = item.get("lfs", {}).get("oid")
    if expected and hashlib.sha256(part.read_bytes()).hexdigest() != expected:
        part.unlink()
        raise ValueError(f"SHA-256 mismatch for {item['path']}")
    part.replace(target)


def _download_source(destination: Path) -> None:
    """Fetch every recording of the pinned repository revision.

    Args:
        destination: Corpus root to fill.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a file fails its checksum.
    """
    repo = str(_CFG["repo_id"])
    revision = str(_CFG["revision"])
    listing = (
        f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true"
    )
    with urllib.request.urlopen(listing) as response:
        items = json.load(response)
    wavs = [i for i in items if i["type"] == "file" and i["path"].endswith(".wav")]
    todo = [i for i in wavs if not (destination / i["path"]).is_file()]
    logger.info("Downloading %d of %d recordings", len(todo), len(wavs))
    with ThreadPoolExecutor(max_workers=int(_CFG["download_workers"])) as pool:
        list(pool.map(lambda i: _fetch(repo, revision, i, destination), todo))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the folder name.

    Byte-identical recordings are kept once (the first in sorted order).

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` and ``label``.

    Raises:
        ValueError: If a folder name is not in ``emotion_map``.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: dict[str, Path] = {}
    for wav in _list_wavs(source_root):
        emotion = wav.parent.name
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion folder {emotion!r} in {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        digest = hashlib.md5(wav.read_bytes()).hexdigest()
        if _CFG["exclude_duplicates"] and digest in seen:
            logger.warning("%s duplicates %s; skipped", wav, seen[digest])
            continue
        seen[digest] = wav
        entries.append(
            {
                "utt_id": f"thanushs25_tamil-{emotion}-{wav.stem}",
                "source": str(wav),
                "label": _EMOTION_MAP[emotion],
            }
        )
    if not entries:
        raise FileNotFoundError(f"No recordings under {source_root}")
    logger.info("Read %d utterance(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. No speaker information
    exists, so speakers are not held out.

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


class Thanushs25TamilBuilder(DatasetBuilder):
    """Prepare the Thanushs25 Tamil corpus for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one manifest
    per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a source tree is reachable."""
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
        """Download the recordings unless they are already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("Corpus already present; skipping download")
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
