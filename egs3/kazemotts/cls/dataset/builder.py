"""KazEmoTTS dataset builder."""

from __future__ import annotations

import hashlib
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

# `399172782_fear_23989.wav` -> narrator 399172782, emotion fear, utterance 23989.
FILENAME_RE = re.compile(r"^(?P<speaker>\d+)_(?P<emotion>[a-z]+)_(?P<item>\d+)\.wav$")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the KazEmoTTS source tree, in order.

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
    """Return the first candidate holding all KazEmoTTS recordings.

    Recordings are searched recursively below each candidate.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with at least ``expected_files`` recordings below it.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and sum(
                1 for wav in candidate.rglob("*.wav") if FILENAME_RE.match(wav.name)
            )
            >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete KazEmoTTS audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "extracted KazEmoTTS archive, or let `prepare_source` download it "
        "from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$KAZEMOTTS_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_zip(archive: Path, destination: Path) -> None:
    """Extract the WAV files of a zip archive into ``destination``.

    ``espnet3.utils.download_utils`` only extracts gzipped tar archives, so
    this is done here. The ``.txt`` transcripts are not needed; only ``.wav``
    members are written, and each is checked against the destination first:
    ``ZipFile.extract`` does not reject absolute paths or ``..`` components on
    its own.

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
        members = [info for info in zf.infolist() if info.filename.endswith(".wav")]
        for info in members:
            if not (destination / info.filename).resolve().is_relative_to(destination):
                raise ValueError(f"{archive.name} escapes its destination: {info}")
        for info in members:
            zf.extract(info, path=destination)


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
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
    digest = _sha256sum(part)
    if digest != str(_CFG["sha256"]):
        raise ValueError(
            f"{part.name}: sha256 {digest} does not match {_CFG['sha256']}"
        )
    part.rename(archive_path)
    _extract_zip(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the file name.

    Each file sits in ``<narrator>/<train|eval>/``. The narrator folder must
    agree with the file name, and the folder name gives the official split.

    Args:
        source_root: Corpus root with the narrator folders somewhere below it.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker``,
        ``label`` and ``official`` (``train`` or ``test``).

    Raises:
        ValueError: If a file name is unexpected, an emotion or split folder is
            unknown, the narrator folder disagrees with the file name, or two
            files get the same utterance id.
        FileNotFoundError: If no recording is found.
    """
    official = {str(k): str(v) for k, v in _CFG["official_splits"].items()}
    entries = []
    seen: dict[str, Path] = {}
    for wav in sorted(source_root.rglob("*.wav")):
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            raise ValueError(f"Unexpected file name: {wav}")
        if fields["emotion"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {fields['emotion']!r} in {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        if wav.parent.name not in official:
            raise ValueError(f"{wav} is not in a {sorted(official)} folder")
        if wav.parent.parent.name != fields["speaker"]:
            raise ValueError(f"{wav} does not sit in its narrator's folder")
        utt_id = f"kazemotts-{wav.stem}"
        if utt_id in seen:
            raise ValueError(f"Duplicate utterance id: {wav} and {seen[utt_id]}")
        seen[utt_id] = wav
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "speaker": fields["speaker"],
                "label": _EMOTION_MAP[fields["emotion"]],
                "official": official[wav.parent.name],
            }
        )
    if not entries:
        raise FileNotFoundError(f"No KazEmoTTS audio under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Keep the official train/test split and draw the validation set from train.

    The recordings in the ``eval`` folders are the test set. A ``valid_ratio``
    share of the ``train`` recordings of each label, shuffled with
    ``split_seed``, becomes the validation set. Narrators are not held out.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If ``valid_ratio`` is not in [0, 1).
    """
    valid_ratio = float(_CFG["valid_ratio"])
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError(f"`valid_ratio` must be in [0, 1), got {valid_ratio}")

    rng = Random(int(_CFG["split_seed"]))
    assignment: dict[str, str] = {}
    train_by_label: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        if entry["official"] == "test":
            assignment[entry["utt_id"]] = "test"
        else:
            train_by_label.setdefault(entry["label"], []).append(entry["utt_id"])

    for _, utt_ids in sorted(train_by_label.items()):
        shuffled = list(utt_ids)
        rng.shuffle(shuffled)
        n_valid = round(len(shuffled) * valid_ratio)
        for utt_id in shuffled[:n_valid]:
            assignment[utt_id] = "valid"
        for utt_id in shuffled[n_valid:]:
            assignment[utt_id] = "train"
    return assignment


class KazEmoTTSBuilder(DatasetBuilder):
    """Prepare KazEmoTTS for the classification recipe.

    The release is 22.05 kHz mono 16-bit WAV, so `build()` resamples every
    recording to 16 kHz and writes one manifest per split. The test set is the
    official ``eval`` folders; validation is drawn from ``train`` (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a KazEmoTTS source tree is reachable."""
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
            logger.info("KazEmoTTS already present; skipping download")
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
