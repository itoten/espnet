"""EmoVoice-DB dataset builder (Hugging Face files, official split)."""

from __future__ import annotations

import hashlib
import json
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


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the EmoVoice-DB files, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate source roots, most specific first.
    """
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def _is_complete(candidate: Path) -> bool:
    """Tell whether the split lists and enough recordings are present."""
    if not all((candidate / name).is_file() for name in _CFG["splits"].values()):
        return False
    count = sum(1 for _ in candidate.glob("*/*.wav"))
    return count >= int(_CFG["expected_files"])


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the split lists and the recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Root with ``train.jsonl``, ``val.jsonl``, ``test.jsonl`` and
        ``<emotion>/<key>.wav`` below it.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir() and _is_complete(candidate):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete EmoVoice-DB files not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the split "
        "lists and the extracted emotion folders, or let `prepare_source` "
        "download them from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMOVOICEDB_OUTPUT`` when set, else ``<recipe_dir>/data``.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_source(destination: Path) -> None:
    """Fetch the split lists and emotion archives, check them and unpack.

    Args:
        destination: Directory to download and unpack into.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a SHA-256 digest does not match.
        zipfile.BadZipFile: If an archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    base_url = str(_CFG["base_url"])
    for name, sha256 in _CFG["files"].items():
        path = destination / str(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        download_url(f"{base_url}/{name}", part, logger=logger)
        digest = _sha256(part)
        if digest != str(sha256):
            raise ValueError(f"SHA-256 mismatch for {name}: {digest}")
        part.rename(path)
        if path.suffix == ".zip":
            _extract_wavs(path, destination)
            if _CFG.get("remove_archive", False):
                path.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the utterances of the three official lists.

    ``train.jsonl`` repeats each recording once per paraphrased emotion
    description, so rows are de-duplicated by ``key``.

    Args:
        source_root: Root with the split lists and the emotion folders.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker``,
        ``label`` and ``split``.

    Raises:
        ValueError: If an emotion is unknown, a key changes emotion or occurs
            in two splits.
        FileNotFoundError: If a listed recording is missing.
    """
    entries: dict[str, dict[str, str]] = {}
    for split, list_name in _CFG["splits"].items():
        with (source_root / list_name).open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                key, emotion = row["key"], row["emotion"]
                if emotion not in _EMOTION_MAP:
                    raise ValueError(
                        f"Unknown emotion {emotion!r} for {key}. "
                        f"Known: {sorted(_EMOTION_MAP)}"
                    )
                previous = entries.get(key)
                if previous is not None:
                    if previous["split"] != split or previous["emotion"] != emotion:
                        raise ValueError(f"{key} is inconsistent across rows")
                    continue
                wav = source_root / emotion / f"{key}.wav"
                if not wav.is_file():
                    raise FileNotFoundError(f"Listed recording missing: {wav}")
                entries[key] = {
                    "utt_id": f"emovoicedb-{key}",
                    "source": str(wav),
                    "speaker": key.rsplit("_", 1)[1],
                    "emotion": emotion,
                    "label": _EMOTION_MAP[emotion],
                    "split": split,
                }
    logger.info(
        "Read %d utterance(s) with %d voice(s)",
        len(entries),
        len({e["speaker"] for e in entries.values()}),
    )
    return sorted(entries.values(), key=lambda e: e["utt_id"])


class EmovoicedbBuilder(DatasetBuilder):
    """Prepare EmoVoice-DB for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one manifest
    per split, following the official train/validation/test lists.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the EmoVoice-DB files are reachable."""
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
            source_dir: Explicit location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the files are still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("EmoVoice-DB already present; skipping download")
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
            source_dir: Explicit location of the source files.

        Raises:
            FileNotFoundError: If the files or ``ffmpeg`` cannot be found.
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

        with ThreadPoolExecutor(max_workers=int(_CFG["convert_workers"])) as pool:
            wavs = list(pool.map(convert, entries))

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry, wav in zip(entries, wavs):
            rows[entry["split"]].append(
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
