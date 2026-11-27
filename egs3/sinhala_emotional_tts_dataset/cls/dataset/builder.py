"""sinhala-emotional-tts-dataset builder (one label folder per class)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
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
_LABEL_FOLDERS = {str(k): str(v) for k, v in _CFG["label_folders"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the dataset, in order.

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


def list_audio(root: Path) -> list[Path]:
    """Return the audio files in the label folders of ``root``, sorted.

    A file belongs to the folder it is in; any file whose name contains one of
    ``audio_extensions`` (the release doubles the extension, e.g.
    ``01_01_ang.wav.wav``, and has one ``.wav.m4a``) is taken.
    """
    exts = tuple(str(e) for e in _CFG["audio_extensions"])
    files = []
    for folder in _LABEL_FOLDERS:
        d = root / folder
        if d.is_dir():
            files += [p for p in d.iterdir() if p.is_file() and p.name.endswith(exts)]
    return sorted(files)


def _is_complete(root: Path) -> bool:
    """Return whether ``root`` holds the expected number of audio files."""
    return len(list_audio(root)) == int(_CFG["expected_files"])


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding all the expected audio files.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Source root holding the label folders.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if _is_complete(candidate):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete sinhala-emotional-tts-dataset not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"the label folders {sorted(_LABEL_FOLDERS)}, or let "
        "`prepare_source` download them from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMO_EMILIA_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _download_source(destination: Path) -> None:
    """Fetch the label folders from Hugging Face.

    The pinned revision is used. Files already present in ``destination``
    are not fetched again, so an interrupted download resumes.

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


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings and label each by its folder.

    Args:
        source_root: Source root (see :func:`resolve_source_root`).

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` and ``label``.

    Raises:
        ValueError: If an utterance id repeats.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: set[str] = set()
    for path in list_audio(source_root):
        stem = path.name.split(".", 1)[0]
        utt_id = f"sinhala_emotional_tts_dataset-{stem}"
        if utt_id in seen:
            raise ValueError(f"Duplicate utterance id: {utt_id}")
        seen.add(utt_id)
        entries.append(
            {"utt_id": utt_id, "source": str(path), "label": _LABEL_FOLDERS[path.parent.name]}
        )
    if not entries:
        raise FileNotFoundError(f"No recording found under {source_root}")
    logger.info("Read %d recording(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the clips at random within each label.

    Each label's clips are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. Speakers are not known,
    so they are not held out.

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


def _convert(ffmpeg: str, source: str, wav: Path, sampling_rate: str) -> None:
    """Write one clip as 16 kHz mono 16-bit PCM."""
    part = wav.with_suffix(wav.suffix + ".part")
    subprocess.run(
        [ffmpeg, "-i", source, "-ac", "1", "-ar", sampling_rate]
        + ["-c:a", "pcm_s16le", "-f", "wav", "-y", "-hide_banner"]
        + ["-loglevel", "error", str(part)],
        check=True,
    )
    part.replace(wav)


class SinhalaEmotionalTTSBuilder(DatasetBuilder):
    """Prepare sinhala-emotional-tts-dataset for the classification recipe.

    `build()` converts every clip to 16 kHz mono and writes one manifest per
    split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete source is reachable."""
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
            logger.info("Source already present; skipping download")
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
        """Convert the clips and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location.
            num_workers: ffmpeg processes run at the same time.

        Raises:
            FileNotFoundError: If the source or ``ffmpeg`` cannot be found.
            ValueError: If an emotion is not listed in ``emotion_map``.
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
        assignment = assign_splits(entries)

        todo = [
            (e["source"], wav_dir / f"{e['utt_id']}.wav")
            for e in entries
            if not (wav_dir / f"{e['utt_id']}.wav").is_file()
        ]
        logger.info("Converting %d of %d clip(s)", len(todo), len(entries))
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            list(pool.map(lambda t: _convert(ffmpeg, t[0], t[1], sampling_rate), todo))

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
