"""nEMO dataset builder."""

from __future__ import annotations

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
_SHARD_PATHS = [str(p) for p in _CFG["shard_paths"]]


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the nEMO parquet shards, in order.

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
    """Return the first candidate holding every parquet shard.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with ``shard_paths`` below it.

    Raises:
        FileNotFoundError: If no candidate holds them all.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if all((candidate / shard).is_file() for shard in _SHARD_PATHS):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "nEMO parquet shards not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        + ", ".join(_SHARD_PATHS)
        + ", or let `prepare_source` download them from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$NEMO_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _download_source(destination: Path) -> None:
    """Fetch every parquet shard at the pinned revision.

    Args:
        destination: Directory to write the shards into.

    Raises:
        urllib.error.URLError: If a download fails.
    """
    base = f"{str(_CFG['repo_url']).rstrip('/')}/{_CFG['revision']}"
    for shard in _SHARD_PATHS:
        target = destination / shard
        if target.is_file():
            continue
        part = target.with_suffix(target.suffix + ".part")
        download_url(f"{base}/{shard}", part, logger=logger)
        part.rename(target)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect one row per recording from the shards, without the audio.

    Args:
        source_root: Corpus root holding the shards.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``speaker`` and ``label``.

    Raises:
        ValueError: If a row carries an unknown emotion, or two rows share a
            ``file_id``.
    """
    entries = []
    seen: set[str] = set()
    for shard in _SHARD_PATHS:
        table = pq.read_table(
            source_root / shard, columns=["file_id", "emotion", "speaker_id"]
        )
        for row in table.to_pylist():
            if row["emotion"] not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {row['emotion']!r} for {row['file_id']}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            if row["file_id"] in seen:
                raise ValueError(f"Duplicate file_id: {row['file_id']}")
            seen.add(row["file_id"])
            entries.append(
                {
                    "utt_id": f"nemo-{row['file_id']}",
                    "speaker": row["speaker_id"],
                    "label": _EMOTION_MAP[row["emotion"]],
                }
            )
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. Speakers are not held out,
    so the splits share speakers.

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


class NEMOBuilder(DatasetBuilder):
    """Prepare nEMO for the classification recipe.

    The release stores 24 kHz mono WAV inside parquet shards, so `build()`
    pipes each recording through ffmpeg to a 16 kHz WAV file and writes one
    manifest per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether every nEMO parquet shard is reachable."""
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
        """Download the parquet shards unless they are already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If a shard is still missing afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("nEMO already present; skipping download")
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
            FileNotFoundError: If the shards or ``ffmpeg`` cannot be found.
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

        # Read one row group at a time: a shard holds ~265 MB of audio.
        for shard in _SHARD_PATHS:
            parquet = pq.ParquetFile(source_root / shard)
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=["file_id", "audio"])
                for row in table.to_pylist():
                    wav = wav_dir / f"nemo-{row['file_id']}.wav"
                    if wav.is_file():
                        continue
                    part = wav.with_suffix(wav.suffix + ".part")
                    subprocess.run(
                        [
                            ffmpeg,
                            "-i",
                            "pipe:0",
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
                        input=row["audio"]["bytes"],
                        check=True,
                    )
                    part.replace(wav)

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            wav = wav_dir / f"{entry['utt_id']}.wav"
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
