"""Saraiki speech emotion dataset builder (manually extracted Kaggle data)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# `Angry/Angry_001.opus`: the folder name is the emotion.
FILENAME_RE = re.compile(r"^[A-Za-z]+_\d+\.opus$")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the Saraiki source tree, in order.

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


def _count_files(root: Path) -> int:
    return sum(
        1
        for folder in _EMOTION_MAP
        for f in (root / folder).glob("*.opus")
        if FILENAME_RE.match(f.name)
    )


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding ``Angry/``, ``Fear/``, ``Happy/``,
        ``Neutral/``, ``Sad/`` and ``Shock/``.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir() and _count_files(candidate) >= _CFG["expected_files"]:
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Saraiki audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nDownload and unzip {_CFG['kaggle_url']} (free Kaggle account), then "
        f"set ${_CFG['source_env_var']} to the directory holding the "
        "`Angry/`, `Fear/`, `Happy/`, `Neutral/`, `Sad/` and `Shock/` folders."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$SARAIKI_SER_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _drop_duplicates(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Leave out the utterances listed in ``duplicate_ids`` of the config.

    Applies only when ``exclude_duplicates`` is true. Each listed utterance has
    the same sound and label as one that stays.

    Raises:
        ValueError: If a listed id is not among ``entries``.
    """
    if not _CFG["exclude_duplicates"]:
        return entries
    drop = {str(i) for i in _CFG["duplicate_ids"]}
    kept = [e for e in entries if e["utt_id"] not in drop]
    missing = drop - {e["utt_id"] for e in entries}
    if missing:
        raise ValueError(f"`duplicate_ids` not found: {sorted(missing)}")
    return kept


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the folder name.

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` and ``label``.

    Raises:
        FileNotFoundError: If no recording is found.
    """
    entries = []
    for folder, label in sorted(_EMOTION_MAP.items()):
        for opus in sorted((source_root / folder).glob("*.opus")):
            if not FILENAME_RE.match(opus.name):
                logger.warning("Skipped unexpected file name: %s", opus)
                continue
            entries.append(
                {
                    "utt_id": f"saraiki_ser-{opus.stem.lower()}",
                    "source": str(opus),
                    "label": label,
                }
            )
    if not entries:
        raise FileNotFoundError(f"No Saraiki audio under {source_root}")
    logger.info("Read %d utterance(s)", len(entries))
    return _drop_duplicates(entries)


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. Speaker identity is not
    published, so speakers cannot be held out.

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


class SaraikiSerBuilder(DatasetBuilder):
    """Prepare Saraiki for the classification recipe.

    `build()` resamples every recording from 48 kHz opus to 16 kHz mono and writes
    one manifest per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a Saraiki source tree is reachable."""
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
        """Check that the manually extracted corpus is available.

        Raises:
            FileNotFoundError: With the Kaggle URL if the tree is missing.
        """
        resolve_source_root(Path(recipe_dir).resolve(), source_dir)

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
