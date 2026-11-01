"""WHiSER dataset builder (8 classes, files + consensus label table)."""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import subprocess
import urllib.request
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
_EMOTION_CODES = {str(k): str(v) for k, v in _CFG["emotion_codes"].items()}
_DROP_CODES = {str(c) for c in _CFG["drop_codes"]}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the WHiSER source tree, in order.

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


def missing_source_entries(root: Path) -> list[str]:
    """Return the corpus entries that are absent or incomplete under ``root``."""
    missing = []
    labels = root / str(_CFG["labels_csv"])
    if not labels.is_file():
        missing.append(str(labels))
    audio_dir = root / str(_CFG["audio_subdir"])
    n_wav = sum(1 for _ in audio_dir.glob("*.wav")) if audio_dir.is_dir() else 0
    if n_wav < int(_CFG["expected_files"]):
        missing.append(f"{audio_dir} ({n_wav}/{_CFG['expected_files']} wav files)")
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the label table and every recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding ``labels_consensus.csv`` and ``wav/``.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        missing = missing_source_entries(candidate)
        if not missing:
            return candidate
        checked.append(f"{candidate} (missing: {', '.join(missing)})")
    raise FileNotFoundError(
        "Complete WHiSER corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['labels_csv']}` and `{_CFG['audio_subdir']}/`, or let "
        "`prepare_source` download them."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$WHISER_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _fetch(url: str, path: Path) -> None:
    """Download ``url`` to ``path`` through a ``.part`` file."""
    part = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, part)
    part.replace(path)


def _download_source(destination: Path) -> None:
    """Fetch the label table and every recording listed on the audio page.

    Args:
        destination: Corpus root to fill.

    Raises:
        urllib.error.URLError: If a download fails.
        RuntimeError: If the listing names fewer than ``expected_files``
            recordings.
    """
    audio_dir = destination / str(_CFG["audio_subdir"])
    audio_dir.mkdir(parents=True, exist_ok=True)

    labels = destination / str(_CFG["labels_csv"])
    if not labels.is_file():
        _fetch(str(_CFG["labels_url"]), labels)

    base = str(_CFG["audio_url"])
    with urllib.request.urlopen(base) as response:
        page = response.read().decode("utf-8", errors="replace")
    names = sorted(set(re.findall(r'href="([^"?/]+\.wav)"', page)))
    if len(names) < int(_CFG["expected_files"]):
        raise RuntimeError(
            f"{base} lists {len(names)} wav files, expected {_CFG['expected_files']}"
        )

    todo = [n for n in names if not (audio_dir / n).is_file()]
    logger.info("Downloading %d of %d recordings", len(todo), len(names))
    with ThreadPoolExecutor(max_workers=int(_CFG["download_workers"])) as pool:
        for i, _ in enumerate(
            pool.map(lambda n: _fetch(base + n, audio_dir / n), todo), 1
        ):
            if i % 500 == 0:
                logger.info("Downloaded %d / %d", i, len(todo))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the consensus table into one row per usable recording.

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``group`` and
        ``label``. ``group`` is the file-name prefix before the first ``.``.

    Raises:
        ValueError: If a row carries an emotion code that is neither kept nor
            listed for dropping, or an `exclude_files` or `duplicate_files` name
            is absent.
        FileNotFoundError: If a recording named in the table is absent.
    """
    audio_dir = source_root / str(_CFG["audio_subdir"])
    entries = []
    dropped: dict[str, int] = {}
    excluded = set(_CFG["exclude_files"])
    if _CFG["exclude_duplicates"]:
        excluded |= set(_CFG["duplicate_files"])
    with (source_root / str(_CFG["labels_csv"])).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            name = row["FileName"].strip()
            code = row["EmoClass"].strip()
            if name in excluded:
                excluded.discard(name)
                continue
            if code in _DROP_CODES:
                dropped[code] = dropped.get(code, 0) + 1
                continue
            if code not in _EMOTION_CODES:
                raise ValueError(
                    f"Unknown emotion code {code!r} for {name}. "
                    f"Known: {sorted(_EMOTION_CODES)}; dropped: {sorted(_DROP_CODES)}"
                )
            wav = audio_dir / name
            if not wav.is_file():
                raise FileNotFoundError(
                    f"Recording named in the table is absent: {wav}"
                )
            entries.append(
                {
                    "utt_id": f"whiser-{Path(name).stem}",
                    "source": str(wav),
                    "group": name.split(".")[0],
                    "label": _EMOTION_CODES[code],
                }
            )
    if excluded:
        raise ValueError(f"`exclude_files` names not found: {sorted(excluded)}")
    if dropped:
        logger.info(
            "Dropped %d recording(s) by emotion code (%s); see dataset/config.yaml",
            sum(dropped.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())),
        )
    logger.info(
        "Read %d utterance(s) in %d group(s)",
        len(entries),
        len({e["group"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split whole groups at random so no group spans two splits.

    Groups are shuffled with ``split_seed`` and filled into test, then valid,
    then train until each holds about its ratio of the utterances. Labels are
    not balanced across splits.

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

    by_group: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        by_group.setdefault(entry["group"], []).append(entry["utt_id"])
    groups = sorted(by_group)
    Random(int(_CFG["split_seed"])).shuffle(groups)

    total = len(entries)
    quota = {"test": test_ratio * total, "valid": valid_ratio * total}
    counts = {"test": 0, "valid": 0}
    assignment: dict[str, str] = {}
    for group in groups:
        for split in ("test", "valid"):
            if counts[split] < quota[split]:
                break
        else:
            split = "train"
        if split != "train":
            counts[split] += len(by_group[group])
        for utt_id in by_group[group]:
            assignment[utt_id] = split
    return assignment


class WhiserBuilder(DatasetBuilder):
    """Prepare WHiSER for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one manifest
    per split, drawn as a random split of whole groups (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a WHiSER source tree is reachable."""
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
        """Download the label table and recordings unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("WHiSER already present; skipping download")
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
