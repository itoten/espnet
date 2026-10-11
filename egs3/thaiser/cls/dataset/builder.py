"""Thai SER dataset builder."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `s001_con_actor001_impro1_1.flac`: recording id, microphone, actor, take.
FILENAME_RE = re.compile(r"^(?P<recording>[sz]\d{3}_[a-z]+)_actor(?P<speaker>\d{3})_")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the Thai SER source tree, in order.

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


def load_labels(label_path: Path) -> dict[str, tuple[str, str]]:
    """Read the label file and keep the recordings that are usable.

    A recording is kept when annotators reached a majority on one of the
    labels in ``emotion_map`` and ``agreement`` is at least ``min_agreement``.

    Args:
        label_path: ``emotion_label.json`` of the release.

    Returns:
        dict[str, tuple[str, str]]: File name to ``(speaker, label)``.

    Raises:
        ValueError: If a file name is unexpected or the number of kept
            recordings differs from ``expected_files``.
    """
    with label_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    labels = {}
    for name, votes in raw.items():
        annotation = votes[0]
        emotion = annotation["majority_emo"]
        if emotion not in _EMOTION_MAP:
            continue
        if annotation["agreement"] < float(_CFG["min_agreement"]):
            continue
        fields = FILENAME_RE.match(name)
        if fields is None:
            raise ValueError(f"Unexpected recording name in {label_path}: {name}")
        labels[name] = (fields["speaker"], _EMOTION_MAP[emotion])
    if len(labels) != _CFG["expected_files"]:
        raise ValueError(
            f"{label_path} keeps {len(labels)} recordings, expected "
            f"{_CFG['expected_files']}"
        )
    return labels


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the label file and all its recordings.

    Recordings are searched recursively below each candidate.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with every usable recording below it.

    Raises:
        FileNotFoundError: If no candidate holds the label file and every
            recording kept by :func:`load_labels`.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        label_path = candidate / str(_CFG["label_file"])
        if label_path.is_file():
            labels = load_labels(label_path)
            found = {flac.name for flac in candidate.rglob("*.flac")}
            if labels.keys() <= found:
                return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete Thai SER audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"{_CFG['label_file']} and the extracted archives, or let "
        "`prepare_source` download them from the GitHub release."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$THAISER_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _fetch(name: str, size: int, destination: Path) -> Path:
    """Download a release file and check its size.

    Args:
        name: File name in the release.
        size: Expected size in bytes.
        destination: Directory to save into.

    Returns:
        Path: The downloaded file.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the downloaded size differs from ``size``.
    """
    path = destination / name
    part = path.with_suffix(path.suffix + ".part")
    download_url(f"{_CFG['release_url']}/{name}", part, logger=logger)
    if part.stat().st_size != size:
        raise ValueError(f"{name} is {part.stat().st_size} bytes, expected {size}")
    part.rename(path)
    return path


def _extract_zip(archive: Path, destination: Path, wanted: set[str]) -> None:
    """Extract the wanted recordings of a zip archive into ``destination``.

    ``espnet3.utils.download_utils`` only extracts gzipped tar archives, so
    this is done here. Each archive holds the recordings of every microphone;
    only members whose file name is in ``wanted`` are written, and each is
    checked against the destination first: ``ZipFile.extract`` does not reject
    absolute paths or ``..`` components on its own.

    Args:
        archive: Archive to read.
        destination: Directory to extract into.
        wanted: File names to keep.

    Raises:
        ValueError: If a member would be written outside ``destination``.
        zipfile.BadZipFile: If the archive is invalid.
    """
    logger.info("Extracting %s", archive.name)
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = [
            info
            for info in zf.infolist()
            if Path(info.filename).name in wanted
            and "__MACOSX" not in info.filename.split("/")
        ]
        for info in members:
            if not (destination / info.filename).resolve().is_relative_to(destination):
                raise ValueError(f"{archive.name} escapes its destination: {info}")
        for info in members:
            zf.extract(info, path=destination)


def _download_source(destination: Path) -> None:
    """Fetch the label file and the archives, then unpack the usable recordings.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a download has an unexpected size.
        zipfile.BadZipFile: If an archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    label_path = _fetch(
        str(_CFG["label_file"]), int(_CFG["label_file_size"]), destination
    )
    wanted = set(load_labels(label_path))
    for name, size in _CFG["archives"].items():
        archive = _fetch(str(name), int(size), destination)
        _extract_zip(archive, destination, wanted)
        if _CFG.get("remove_archives", False):
            archive.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the usable recordings, taking labels from the label file.

    Args:
        source_root: Corpus root holding the label file and the recordings.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If the label file does not keep ``expected_files``
            recordings or a recording name occurs twice.
        FileNotFoundError: If a usable recording is missing.
    """
    labels = load_labels(source_root / str(_CFG["label_file"]))
    sources: dict[str, Path] = {}
    for flac in sorted(source_root.rglob("*.flac")):
        if flac.name not in labels:
            continue
        if flac.name in sources:
            raise ValueError(f"Duplicate recording: {flac} and {sources[flac.name]}")
        sources[flac.name] = flac
    missing = labels.keys() - sources.keys()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} recordings missing, e.g. {min(missing)}"
        )
    entries = [
        {
            "utt_id": f"thaiser-{Path(name).stem}",
            "source": str(sources[name]),
            "speaker": speaker,
            "label": label,
        }
        for name, (speaker, label) in sorted(labels.items())
    ]
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. Speakers are not held out, so the
    splits share speakers.

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


class ThaiSERBuilder(DatasetBuilder):
    """Prepare Thai SER for the classification recipe.

    The release holds 44.1 kHz FLAC; `build()` converts the recordings kept by
    :func:`load_labels` to 16 kHz mono WAV with ffmpeg and writes one manifest
    per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the Thai SER labels and recordings are reachable."""
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
            logger.info("Thai SER already present; skipping download")
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

        Recordings are converted by ``num_workers`` ffmpeg processes at a time.

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

        def convert(entry: dict[str, str]) -> Path:
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
            return wav

        with ThreadPoolExecutor(max_workers=int(_CFG["num_workers"])) as pool:
            wavs = list(pool.map(convert, entries))
        for entry, wav in zip(entries, wavs):
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
