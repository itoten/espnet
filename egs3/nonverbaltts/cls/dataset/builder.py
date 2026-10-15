"""NonverbalTTS dataset builder (English speech with emotion tags, 7 classes)."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import urllib.request
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}
_SPLIT_MAP = {str(k): str(v) for k, v in _CFG["split_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the NonverbalTTS parquet files.

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


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate that holds every listed parquet file.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Source root holding ``default/<split>/*.parquet``.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if all((candidate / str(item["path"])).is_file() for item in _CFG["files"]):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete NonverbalTTS corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "`default/<split>/*.parquet` files, or let `prepare_source` download "
        "them from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$NONVERBALTTS_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_source(destination: Path) -> None:
    """Fetch every listed parquet file and check its SHA-256.

    Args:
        destination: Source root to download into.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a checksum does not match.
    """
    base_url = str(_CFG["base_url"]).rstrip("/")
    for item in _CFG["files"]:
        target = destination / str(item["path"])
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        logger.info("Downloading %s", item["path"])
        request = urllib.request.Request(
            f"{base_url}/{item['path']}", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out)
        digest = _sha256(part)
        if digest != str(item["sha256"]):
            raise ValueError(f"SHA-256 mismatch for {item['path']}: {digest}")
        part.rename(target)


def _to_wav(ffmpeg: str, audio: bytes, output: Path, sampling_rate: str) -> None:
    """Convert encoded audio bytes to a 16 kHz mono wav file via ffmpeg."""
    part = output.with_suffix(output.suffix + ".part")
    subprocess.run(
        [ffmpeg, "-i", "pipe:0", "-ac", "1", "-ar", sampling_rate, "-f", "wav"]
        + ["-y", "-hide_banner", "-loglevel", "error", str(part)],
        input=audio,
        check=True,
    )
    part.replace(output)


class NonverbalttsBuilder(DatasetBuilder):
    """Prepare NonverbalTTS for the classification recipe.

    `build()` decodes the audio stored in the parquet files, resamples it to
    16 kHz mono and writes one manifest per split. The published ``train``,
    ``dev`` and ``test`` splits are used; ``other`` is not downloaded.
    Utterances without an emotion tag or tagged ``other`` are dropped and
    counted in the log.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the NonverbalTTS parquet files are reachable."""
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
        """Download the parquet files unless they are already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the files are still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("NonverbalTTS already present; skipping download")
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
        """Decode the audio and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location.

        Raises:
            FileNotFoundError: If the parquet files or ``ffmpeg`` cannot be found.
            ValueError: If an emotion tag is not listed in ``emotion_map``.
            subprocess.CalledProcessError: If a conversion fails.
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

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        dropped: Counter[str] = Counter()
        seen: set[str] = set()
        for item in _CFG["files"]:
            published_split = Path(str(item["path"])).parent.name
            split = _SPLIT_MAP[published_split]
            parquet = pq.ParquetFile(source_root / str(item["path"]))
            for batch in parquet.iter_batches(
                batch_size=32, columns=["index", "audio", "Emotion"]
            ):
                for row in batch.to_pylist():
                    emotion = row["Emotion"]
                    if emotion is None or emotion in _CFG["drop_emotions"]:
                        dropped[f"{split}:{emotion}"] += 1
                        continue
                    if emotion not in _EMOTION_MAP:
                        raise ValueError(
                            f"Unknown emotion {emotion!r} for {row['index']}. "
                            f"Known: {sorted(_EMOTION_MAP)}"
                        )
                    utt_id = "nonverbaltts-" + row["index"].replace("/", "_")
                    if utt_id in seen:
                        raise ValueError(f"Duplicate utterance id: {utt_id}")
                    seen.add(utt_id)
                    wav = wav_dir / f"{utt_id}.wav"
                    if not wav.is_file():
                        _to_wav(ffmpeg, row["audio"]["bytes"], wav, sampling_rate)
                    rows[split].append(
                        (utt_id, str(wav.resolve()), _EMOTION_MAP[emotion])
                    )

        for key, count in sorted(dropped.items()):
            logger.info("Dropped %d utterance(s) with emotion %s", count, key)
        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
