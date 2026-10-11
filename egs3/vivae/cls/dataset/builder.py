"""VIVAE dataset builder."""

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
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `S03_pain_peak_02.wav`: speaker, emotion word, intensity, recording number.
FILENAME_RE = re.compile(
    r"^(?P<speaker>S\d{2})_(?P<emotion>[a-z]+)_(?P<intensity>[a-z]+)_"
    r"(?P<number>\d+)\.wav$"
)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}
_RECORDINGS_DIR = Path(str(_CFG["recordings_dir"]))


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the VIVAE source tree, in order.

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
    """Return the first candidate holding all VIVAE recordings.

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
        if (
            candidate.is_dir()
            and sum(1 for _ in (candidate / _RECORDINGS_DIR).glob("*.wav"))
            >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete VIVAE audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "extracted VIVAE archive, or let `prepare_source` download it "
        "from Zenodo."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$VIVAE_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_zip(archive: Path, destination: Path) -> None:
    """Extract the ``full_set`` WAV files of a zip archive into ``destination``.

    ``espnet3.utils.download_utils`` only extracts gzipped tar archives, so
    this is done here. The archive also holds ``core_set`` (a subset of
    ``full_set``) and macOS metadata; only the ``recordings_dir`` ``.wav``
    members are written, and each is checked against the destination first:
    ``ZipFile.extract`` does not reject absolute paths or ``..`` components
    on its own.

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
        members = [
            info
            for info in zf.infolist()
            if Path(info.filename).parent == _RECORDINGS_DIR
            and info.filename.endswith(".wav")
        ]
        for info in members:
            if not (destination / info.filename).resolve().is_relative_to(destination):
                raise ValueError(f"{archive.name} escapes its destination: {info}")
        for info in members:
            zf.extract(info, path=destination)


def _md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_source(destination: Path) -> None:
    """Fetch and unpack the archive.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the download does not match its published checksum.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / str(_CFG["archive_name"])
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    download_url(str(_CFG["archive_url"]), part, logger=logger)
    digest = _md5sum(part)
    if digest != str(_CFG["md5"]):
        raise ValueError(f"{part.name}: md5 {digest} does not match {_CFG['md5']}")
    part.rename(archive_path)
    _extract_zip(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking speaker and label from the file name.

    Args:
        source_root: Corpus root holding ``recordings_dir``.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a file name does not match the documented pattern, its
            emotion word is not in ``emotion_map``, or two files get the same
            utterance id.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: dict[str, Path] = {}
    for wav in sorted((source_root / _RECORDINGS_DIR).glob("*.wav")):
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            raise ValueError(f"Unexpected file name: {wav}")
        if fields["emotion"] not in _EMOTION_MAP:
            raise ValueError(f"Unknown emotion {fields['emotion']!r} in {wav}")
        utt_id = f"vivae-{wav.stem}"
        if utt_id in seen:
            raise ValueError(f"Duplicate utterance id: {wav} and {seen[utt_id]}")
        seen[utt_id] = wav
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "speaker": fields["speaker"],
                "label": _EMOTION_MAP[fields["emotion"]],
            }
        )
    if not entries:
        raise FileNotFoundError(f"No VIVAE audio under {source_root}")
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


class VIVAEBuilder(DatasetBuilder):
    """Prepare VIVAE for the classification recipe.

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
        """Check whether a VIVAE source tree is reachable."""
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
            logger.info("VIVAE already present; skipping download")
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
