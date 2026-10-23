"""BERSt dataset builder (Hugging Face clips, published speaker-disjoint split)."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import time
import urllib.request
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
_SPLIT_DIRS = {str(k): str(v) for k, v in _CFG["split_dirs"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the BERSt source tree, in order.

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


def _read_metadata(split_dir: Path) -> list[dict[str, str]]:
    """Return the rows of ``metadata.csv`` in ``split_dir`` (empty if absent)."""
    path = split_dir / str(_CFG["metadata_name"])
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def missing_source_entries(root: Path) -> list[str]:
    """Return the clips named in the metadata that are absent under ``root``."""
    missing = []
    for split_dir in _SPLIT_DIRS.values():
        rows = _read_metadata(root / split_dir)
        if not rows:
            missing.append(f"{root / split_dir / _CFG['metadata_name']}")
        missing.extend(
            str(root / split_dir / row["file_name"])
            for row in rows
            if not (root / split_dir / row["file_name"]).is_file()
        )
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every metadata file and clip.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding ``train/``, ``validation/`` and ``test/``, each
        with ``metadata.csv`` and its ``audio/`` clips.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        missing = missing_source_entries(candidate)
        if not missing:
            return candidate
        checked.append(f"{candidate} ({len(missing)} missing, e.g. {missing[0]})")
    raise FileNotFoundError(
        "Complete BERSt corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "`train/`, `validation/` and `test/` clips, or let `prepare_source` "
        "download them from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$BERST_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _fetch(url: str, path: Path) -> None:
    """Download ``url`` to ``path`` through a ``.part`` file, retrying on errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(int(_CFG["download_retries"])):
        try:
            urllib.request.urlretrieve(url, part)
            break
        except OSError:
            if attempt + 1 == int(_CFG["download_retries"]):
                raise
            time.sleep(2 ** (attempt + 1))
    part.replace(path)


def _download_source(destination: Path) -> None:
    """Fetch every metadata file and the clips it lists.

    Args:
        destination: Corpus root to fill.

    Raises:
        urllib.error.URLError: If a download keeps failing.
    """
    base = str(_CFG["base_url"]).rstrip("/") + "/"
    jobs = []
    for split_dir in _SPLIT_DIRS.values():
        meta = destination / split_dir / str(_CFG["metadata_name"])
        if not meta.is_file():
            _fetch(f"{base}{split_dir}/{_CFG['metadata_name']}", meta)
        for row in _read_metadata(destination / split_dir):
            clip = destination / split_dir / row["file_name"]
            if not clip.is_file():
                jobs.append((f"{base}{split_dir}/{row['file_name']}", clip))

    logger.info("Downloading %d clips", len(jobs))
    with ThreadPoolExecutor(max_workers=int(_CFG["download_workers"])) as pool:
        for i, _ in enumerate(pool.map(lambda job: _fetch(*job), jobs), 1):
            if i % 500 == 0:
                logger.info("Downloaded %d / %d", i, len(jobs))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the metadata of every split into one row per clip.

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``split``, ``speaker``
        and ``label``.

    Raises:
        ValueError: If an ``affect`` value is unknown, two clips share an id, or an
            `exclude_files` name is absent.
    """
    entries = []
    seen = set()
    excluded = set(_CFG["exclude_files"])
    for split, split_dir in _SPLIT_DIRS.items():
        for row in _read_metadata(source_root / split_dir):
            affect = row["affect"].strip()
            if affect not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown affect {affect!r} for {row['file_name']}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            # `audio/<recording>/final/<chunk>.wav`
            parts = Path(row["file_name"]).with_suffix("").parts
            utt_id = f"berst-{parts[-3]}-{parts[-1]}"
            if utt_id in seen:
                raise ValueError(f"Duplicate utterance id: {utt_id}")
            seen.add(utt_id)
            if utt_id in excluded:
                excluded.discard(utt_id)
                continue
            entries.append(
                {
                    "utt_id": utt_id,
                    "source": str(source_root / split_dir / row["file_name"]),
                    "split": split,
                    "speaker": row["user_id"],
                    "label": _EMOTION_MAP[affect],
                }
            )
    if excluded:
        raise ValueError(f"`exclude_files` names not found: {sorted(excluded)}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


class BerstBuilder(DatasetBuilder):
    """Prepare BERSt for the classification recipe.

    `build()` resamples every clip to 16 kHz mono and writes one manifest per
    split, following the published speaker-disjoint train/validation/test split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a BERSt source tree is reachable."""
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
            logger.info("BERSt already present; skipping download")
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
