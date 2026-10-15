"""EmoTale dataset builder."""

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

# `DK_004_A_5.wav`: Danish (`EN` for English), speaker 004, emotion code A
# (anger), sentence 5.
FILENAME_RE = re.compile(
    r"^(?P<language>DK|EN)_(?P<speaker>\d+)_(?P<emotion>[A-Z])_(?P<sentence>\d)\.wav$"
)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the EmoTale source tree, in order.

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
    table = root / str(_CFG["table_name"])
    if not table.is_file():
        missing.append(str(table))
    audio_dir = root / str(_CFG["audio_subdir"])
    n_wav = sum(1 for _ in audio_dir.glob("*.wav")) if audio_dir.is_dir() else 0
    if n_wav < int(_CFG["expected_files"]):
        missing.append(f"{audio_dir} ({n_wav}/{_CFG['expected_files']} wav files)")
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the table and every recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding ``annotations.csv`` and ``wav/``.

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
        "Complete EmoTale corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['table_name']}` and `{_CFG['audio_subdir']}/`, or let "
        "`prepare_source` download them."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMOTALE_OUTPUT`` when set, else ``<recipe_dir>/data``.
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
    """Fetch the annotation table, the speaker table and every recording.

    The recordings are listed in the annotation table; all files come from one
    pinned commit of the corpus GitHub repository.

    Args:
        destination: Corpus root to fill.

    Raises:
        urllib.error.URLError: If a download fails.
        RuntimeError: If the table lists fewer than ``expected_files`` files.
    """
    base = str(_CFG["base_url"]).rstrip("/") + "/"
    audio_dir = destination / str(_CFG["audio_subdir"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    for name in (str(_CFG["table_name"]), str(_CFG["speaker_table_name"])):
        if not (destination / name).is_file():
            _fetch(base + name, destination / name)

    with (destination / str(_CFG["table_name"])).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        names = sorted({row["file"].strip() for row in csv.DictReader(fh)})
    if len(names) < int(_CFG["expected_files"]):
        raise RuntimeError(
            f"{_CFG['table_name']} lists {len(names)} files, "
            f"expected {_CFG['expected_files']}"
        )

    audio_base = base + str(_CFG["audio_subdir"]) + "/"
    todo = [n for n in names if not (audio_dir / n).is_file()]
    logger.info("Downloading %d of %d recordings", len(todo), len(names))
    with ThreadPoolExecutor(max_workers=int(_CFG["download_workers"])) as pool:
        for i, _ in enumerate(
            pool.map(lambda n: _fetch(audio_base + n, audio_dir / n), todo), 1
        ):
            if i % 200 == 0:
                logger.info("Downloaded %d / %d", i, len(todo))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the file name.

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a file name is malformed or carries an unknown emotion
            code, or if it disagrees with the table's ``gt_emotion``.
        FileNotFoundError: If no recording is found.
    """
    audio_dir = source_root / str(_CFG["audio_subdir"])
    enacted: dict[str, str] = {}
    with (source_root / str(_CFG["table_name"])).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            enacted[row["file"].strip()] = row["gt_emotion"].strip()

    entries = []
    for wav in sorted(audio_dir.glob("*.wav")):
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            raise ValueError(f"Unexpected file name: {wav}")
        if fields["emotion"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion code {fields['emotion']!r} in {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        if enacted.get(wav.name) != fields["emotion"]:
            raise ValueError(
                f"{wav.name}: file-name emotion {fields['emotion']!r} differs from "
                f"the table's {enacted.get(wav.name)!r}"
            )
        entries.append(
            {
                "utt_id": f"emotale-{wav.stem}",
                "source": str(wav),
                "speaker": fields["speaker"],
                "label": _EMOTION_MAP[fields["emotion"]],
            }
        )
    if not entries:
        raise FileNotFoundError(f"No EmoTale audio under {audio_dir}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole speakers: no speaker appears in two splits.

    Speakers are shuffled with ``split_seed`` and cut by ``valid_ratio`` and
    ``test_ratio`` of the speaker count. A speaker id is shared by that
    speaker's Danish and English recordings, which stay together.

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

    speakers = sorted({e["speaker"] for e in entries})
    Random(int(_CFG["split_seed"])).shuffle(speakers)
    n_test = round(len(speakers) * test_ratio)
    n_valid = round(len(speakers) * valid_ratio)
    split_of = {}
    for i, speaker in enumerate(speakers):
        if i < n_test:
            split_of[speaker] = "test"
        elif i < n_test + n_valid:
            split_of[speaker] = "valid"
        else:
            split_of[speaker] = "train"
    return {e["utt_id"]: split_of[e["speaker"]] for e in entries}


class EmotaleBuilder(DatasetBuilder):
    """Prepare EmoTale for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one manifest
    per split, holding out whole speakers (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a EmoTale source tree is reachable."""
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
            logger.info("EmoTale already present; skipping download")
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
