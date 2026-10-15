"""CSEMOTIONS dataset builder (Hugging Face parquet shards, speaker split)."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

import pyarrow.parquet as pq

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
_SHARDS = [(str(s["name"]), str(s["sha256"])) for s in _CFG["shards"]]


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the parquet shards, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate shard directories, most specific first.
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
    """Return the first candidate holding every parquet shard.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding all shards.

    Raises:
        FileNotFoundError: If no candidate holds every shard.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if all((candidate / name).is_file() for name, _ in _SHARDS):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete CSEMOTIONS shards not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the parquet "
        "shards, or let `prepare_source` download them."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$CSEMOTIONS_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_source(destination: Path) -> None:
    """Fetch the shards that are missing and check their SHA-256.

    Args:
        destination: Directory to download into.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a downloaded shard has the wrong SHA-256.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for name, sha256 in _SHARDS:
        target = destination / name
        if target.is_file():
            continue
        part = target.with_suffix(target.suffix + ".part")
        download_url(f"{_CFG['base_url']}/{name}", part, logger=logger)
        digest = _sha256(part)
        if digest != sha256:
            raise ValueError(f"SHA-256 mismatch for {name}: {digest}")
        part.rename(target)


def _to_16k_wav(ffmpeg: str, wav_bytes: bytes, out: Path) -> None:
    """Resample in-memory WAV bytes to a 16 kHz mono file at ``out``."""
    part = out.with_suffix(out.suffix + ".part")
    subprocess.run(
        [ffmpeg, "-i", "pipe:0", "-ac", "1", "-ar", str(_CFG["sampling_rate"])]
        + ["-f", "wav", "-y", "-hide_banner", "-loglevel", "error", str(part)],
        input=wav_bytes,
        check=True,
    )
    part.replace(out)


def assign_splits(speakers_of: dict[str, str]) -> dict[str, str]:
    """Hold out whole speakers: ``n_test_speakers``, ``n_valid_speakers``, rest train.

    Speakers are shuffled with ``split_seed``, so every split has speakers the
    others never see.

    Args:
        speakers_of: Utterance id to speaker.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If fewer speakers exist than the two held-out splits need
            plus one for training.
    """
    n_test = int(_CFG["n_test_speakers"])
    n_valid = int(_CFG["n_valid_speakers"])
    speakers = sorted(set(speakers_of.values()))
    if len(speakers) < n_test + n_valid + 1:
        raise ValueError(f"Only {len(speakers)} speakers for {n_test}+{n_valid}+train")
    Random(int(_CFG["split_seed"])).shuffle(speakers)
    split_of = {s: "test" for s in speakers[:n_test]}
    split_of.update({s: "valid" for s in speakers[n_test : n_test + n_valid]})
    return {utt: split_of.get(spk, "train") for utt, spk in speakers_of.items()}


class CsemotionsBuilder(DatasetBuilder):
    """Prepare CSEMOTIONS for the classification recipe.

    `build()` writes every clip as a 16 kHz mono wav and one manifest per
    split, drawn as a speaker-held-out split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether every parquet shard is reachable."""
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
        """Download the shards unless they are already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit shard location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the shards are still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("CSEMOTIONS already present; skipping download")
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
        """Convert the clips and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit shard location.

        Raises:
            FileNotFoundError: If the shards or ``ffmpeg`` cannot be found.
            ValueError: If an emotion is unknown or a file name repeats.
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

        labels: dict[str, str] = {}
        speakers: dict[str, str] = {}
        for name, _ in _SHARDS:
            reader = pq.ParquetFile(source_root / name)
            for batch in reader.iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    emotion = row["emotion"]
                    if emotion not in _EMOTION_MAP:
                        raise ValueError(
                            f"Unknown emotion {emotion!r}. "
                            f"Known: {sorted(_EMOTION_MAP)}"
                        )
                    utt_id = f"csemotions-{Path(row['audio']['path']).stem}"
                    if utt_id in labels:
                        raise ValueError(f"Duplicate recording name: {utt_id}")
                    labels[utt_id] = _EMOTION_MAP[emotion]
                    speakers[utt_id] = row["speaker"]
                    wav = wav_dir / f"{utt_id}.wav"
                    if not wav.is_file():
                        _to_16k_wav(ffmpeg, row["audio"]["bytes"], wav)
        logger.info(
            "Read %d utterance(s) from %d speaker(s)",
            len(labels),
            len(set(speakers.values())),
        )

        assignment = assign_splits(speakers)
        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for utt_id, label in labels.items():
            wav = wav_dir / f"{utt_id}.wav"
            rows[assignment[utt_id]].append((utt_id, str(wav.resolve()), label))

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
