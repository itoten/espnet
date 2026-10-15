"""ViSEC dataset builder (Hugging Face parquet, speaker-grouped split)."""

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


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the parquet file, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate directories, most specific first.
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
    """Return the first candidate holding the parquet file.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding the parquet file.

    Raises:
        FileNotFoundError: If no candidate holds it.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (candidate / str(_CFG["parquet_name"])).is_file():
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "ViSEC parquet file not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['parquet_name']}`, or let `prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$VISEC_OUTPUT`` when set, else ``<recipe_dir>/data``.
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
    """Fetch the parquet file and check its SHA-256.

    Args:
        destination: Directory to download into.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the file has the wrong SHA-256.
    """
    destination.mkdir(parents=True, exist_ok=True)
    name = str(_CFG["parquet_name"])
    target = destination / name
    part = target.with_suffix(target.suffix + ".part")
    download_url(str(_CFG["parquet_url"]), part, logger=logger)
    digest = _sha256(part)
    if digest != str(_CFG["parquet_sha256"]):
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
    """Keep every speaker group in one split.

    A group larger than the test quota goes to train. The other groups are
    shuffled with ``split_seed`` and fill test, then valid, then train until
    each holds about its ratio of the utterances. Labels are not balanced.

    Args:
        speakers_of: Utterance id to speaker group.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If the ratios leave no room for training.
    """
    valid_ratio = float(_CFG["valid_ratio"])
    test_ratio = float(_CFG["test_ratio"])
    if valid_ratio + test_ratio >= 1.0:
        raise ValueError("`valid_ratio` + `test_ratio` must be below 1")

    by_group: dict[str, list[str]] = {}
    for utt_id in sorted(speakers_of):
        by_group.setdefault(speakers_of[utt_id], []).append(utt_id)

    total = len(speakers_of)
    quota = {"test": test_ratio * total, "valid": valid_ratio * total}
    counts = {"test": 0, "valid": 0}
    assignment: dict[str, str] = {}
    groups = []
    for group, utt_ids in sorted(by_group.items()):
        if len(utt_ids) > quota["test"]:
            assignment.update({u: "train" for u in utt_ids})
        else:
            groups.append(group)
    Random(int(_CFG["split_seed"])).shuffle(groups)
    for group in groups:
        for split in ("test", "valid"):
            if counts[split] < quota[split]:
                break
        else:
            split = "train"
        if split != "train":
            counts[split] += len(by_group[group])
        assignment.update({u: split for u in by_group[group]})
    return assignment


class VisecBuilder(DatasetBuilder):
    """Prepare ViSEC for the classification recipe.

    `build()` writes every distinct clip as a 16 kHz mono wav and one manifest
    per split, drawn as a speaker-grouped split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the parquet file is reachable."""
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
        """Download the parquet file unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the file is still missing afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("ViSEC already present; skipping download")
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

        Byte-identical clips are kept once. If the copies carry different
        labels, none of them is kept.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location of the parquet file.

        Raises:
            FileNotFoundError: If the parquet file or ``ffmpeg`` cannot be found.
            ValueError: If the row count or an emotion is unexpected.
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

        # Audio digest -> (utterance id, label, speaker) of its first copy.
        first: dict[str, tuple[str, str, str]] = {}
        conflicting: set[str] = set()
        n_rows = 0
        reader = pq.ParquetFile(source_root / str(_CFG["parquet_name"]))
        for batch in reader.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                n_rows += 1
                emotion = row["emotion"]
                if emotion not in _EMOTION_MAP:
                    raise ValueError(
                        f"Unknown emotion {emotion!r}. Known: {sorted(_EMOTION_MAP)}"
                    )
                audio = row["path"]["bytes"]
                digest = hashlib.md5(audio).hexdigest()
                label = _EMOTION_MAP[emotion]
                if digest in first:
                    if first[digest][1] != label:
                        conflicting.add(digest)
                    continue
                utt_id = f"visec-{Path(row['path']['path']).stem}"
                first[digest] = (utt_id, label, str(row["speaker_id"]))
                wav = wav_dir / f"{utt_id}.wav"
                if not wav.is_file():
                    _to_16k_wav(ffmpeg, audio, wav)
        if n_rows != int(_CFG["expected_rows"]):
            raise ValueError(f"Read {n_rows} rows, expected {_CFG['expected_rows']}")

        kept = {d: v for d, v in first.items() if d not in conflicting}
        for digest in conflicting:
            (wav_dir / f"{first[digest][0]}.wav").unlink(missing_ok=True)
        logger.info(
            "Read %d row(s): %d distinct clip(s) kept, %d duplicate row(s) "
            "skipped, %d clip(s) dropped for conflicting duplicate labels",
            n_rows,
            len(kept),
            n_rows - len(first),
            len(conflicting),
        )

        labels = {utt: label for utt, label, _ in kept.values()}
        speakers = {utt: spk for utt, _, spk in kept.values()}
        logger.info("%d speaker group(s)", len(set(speakers.values())))
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
