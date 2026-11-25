"""Dialogs dataset builder (expressive conversational Russian speech, 7 emotions)."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
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
_DROP_EMOTIONS = {str(e) for e in _CFG["drop_emotions"]}
_SPLIT_FILES = {str(k): str(v) for k, v in _CFG["split_files"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the Dialogs source, in order.

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


def read_split_file(path: Path) -> list[dict[str, str]]:
    """Read one pipe-delimited split file of the release.

    Quotes are not special in these files (transcripts may contain them), so
    each line is split on ``|`` only.

    Args:
        path: ``train.csv``, ``val.csv`` or ``test.csv``.

    Returns:
        list[dict[str, str]]: One dict per row, keyed by the header.

    Raises:
        ValueError: If a row does not have as many fields as the header.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|", quoting=csv.QUOTE_NONE)
        rows = list(reader)
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed row in {path}: {row}")
    return rows


def count_missing_audio(root: Path) -> int:
    """Count the utterances of the split files whose wav is not under ``root``."""
    return sum(
        not (root / row["audio_path"]).is_file()
        for name in _SPLIT_FILES.values()
        for row in read_split_file(root / name)
    )


def _is_complete(root: Path) -> bool:
    """Return whether ``root`` holds the split files and all the release's audio.

    The release lacks the audio of ``missing_audio_expected`` utterances, so
    the source counts as complete when exactly that many (or none) are
    missing.
    """
    if not all((root / name).is_file() for name in _SPLIT_FILES.values()):
        return False
    return count_missing_audio(root) in (0, int(_CFG["missing_audio_expected"]))


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the split files and all their audio.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Source root holding ``{train,val,test}.csv`` and ``wavs/``.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if _is_complete(candidate):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete Dialogs corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        "`train.csv`, `val.csv`, `test.csv` and `wavs/`, or let "
        "`prepare_source` download them from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$DIALOGS_RU_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _download_source(destination: Path) -> None:
    """Fetch the audio and split files from Hugging Face at the pinned revision.

    Files already present in ``destination`` are not fetched again, so an
    interrupted download resumes.

    Args:
        destination: Source root to download into.
    """
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s into %s", _CFG["hf_repo"], destination)
    snapshot_download(
        str(_CFG["hf_repo"]),
        repo_type="dataset",
        revision=str(_CFG["hf_revision"]),
        allow_patterns=[str(p) for p in _CFG["allow_patterns"]],
        local_dir=destination,
    )


def read_entries(source_root: Path) -> dict[str, list[dict[str, str]]]:
    """Collect the emotion-labelled utterances of each published split.

    Args:
        source_root: Source root (see :func:`resolve_source_root`).

    Returns:
        dict[str, list[dict[str, str]]]: Split name to rows with keys
        ``utt_id``, ``source`` and ``label``.

    Raises:
        ValueError: If a label is neither in ``emotion_map`` nor in
            ``drop_emotions``, or an utterance id repeats.
    """
    entries: dict[str, list[dict[str, str]]] = {}
    dropped: Counter[str] = Counter()
    no_audio: Counter[str] = Counter()
    seen: set[str] = set()
    excluded = (
        {str(i) for i in _CFG["duplicate_ids"]} if _CFG["exclude_duplicates"] else set()
    )
    for split, name in _SPLIT_FILES.items():
        entries[split] = []
        for row in read_split_file(source_root / name):
            if not (source_root / row["audio_path"]).is_file():
                no_audio[split] += 1
                continue
            emotion = row["emotion"].strip()
            if emotion in _DROP_EMOTIONS:
                dropped[f"{split}:{emotion}"] += 1
                continue
            if emotion not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {emotion!r} for {row['audio_path']}. "
                    f"Known: {sorted(_EMOTION_MAP)}; dropped: {sorted(_DROP_EMOTIONS)}"
                )
            utt_id = f"dialogs_ru-{Path(row['audio_path']).stem}"
            if utt_id in seen:
                raise ValueError(f"Duplicate utterance id: {utt_id}")
            seen.add(utt_id)
            if utt_id in excluded:
                dropped[f"{split}:listed duplicate"] += 1
                continue
            entries[split].append(
                {
                    "utt_id": utt_id,
                    "source": str(source_root / row["audio_path"]),
                    "label": _EMOTION_MAP[emotion],
                }
            )
    for key, count in sorted(dropped.items()):
        logger.info("Dropped %d utterance(s) with label %s", count, key)
    for split, count in sorted(no_audio.items()):
        logger.warning("Skipped %d %s utterance(s) without audio in the release", count, split)
    missing = [str(i) for i in _CFG["duplicate_ids"] if str(i) not in seen]
    if missing:
        raise ValueError(f"`duplicate_ids` not found: {missing}")
    return entries


def _convert(ffmpeg: str, source: str, wav: Path, sampling_rate: str) -> None:
    """Write one recording as 16 kHz mono 16-bit PCM."""
    part = wav.with_suffix(wav.suffix + ".part")
    subprocess.run(
        [ffmpeg, "-i", source, "-ac", "1", "-ar", sampling_rate]
        + ["-c:a", "pcm_s16le", "-f", "wav", "-y", "-hide_banner"]
        + ["-loglevel", "error", str(part)],
        check=True,
    )
    part.replace(wav)


class DialogsRuBuilder(DatasetBuilder):
    """Prepare Dialogs for the classification recipe.

    The recordings are 44.1 kHz WAV, so `build()` converts every kept one to
    16 kHz mono and writes one manifest per published split. Utterances with
    a label listed in ``drop_emotions`` (none by default) are left out and
    counted in the log.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete Dialogs source is reachable."""
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
        """Download the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the source is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("Dialogs already present; skipping download")
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
        num_workers: int = 8,
        **_kwargs,
    ) -> None:
        """Convert the recordings and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location.
            num_workers: ffmpeg processes run at the same time.

        Raises:
            FileNotFoundError: If the source or ``ffmpeg`` cannot be found.
            ValueError: If a label is not listed in the config.
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

        entries = read_entries(source_root)
        everything = [e for split_entries in entries.values() for e in split_entries]
        todo = [
            (e["source"], wav_dir / f"{e['utt_id']}.wav")
            for e in everything
            if not (wav_dir / f"{e['utt_id']}.wav").is_file()
        ]
        logger.info("Converting %d of %d recording(s)", len(todo), len(everything))
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            list(pool.map(lambda t: _convert(ffmpeg, t[0], t[1], sampling_rate), todo))

        for split, relpath in _CFG["manifest_paths"].items():
            rows = [
                (e["utt_id"], str((wav_dir / f"{e['utt_id']}.wav").resolve()),
                 e["label"])
                for e in entries[split]
            ]
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows), manifest)
