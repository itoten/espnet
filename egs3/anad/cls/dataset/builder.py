"""ANAD dataset builder (3 classes, call/turn-grouped split).

Reads `ANAD.csv` and the 1-second wav chunks, keeps the chunks that have both a
label and a file, and resamples them to 16 kHz mono into `$ANAD_OUTPUT/wav`.
"""

from __future__ import annotations

import csv
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

# `V7_1 (145).wav` or `V4-8 (3).wav`: call 7, turn 1, chunk 145.
NAME_RE = re.compile(r"^V(?P<call>\d+)[_-](?P<turn>\d+) \((?P<chunk>\d+)\)\.wav$")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the corpus tree, in order."""
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def missing_source_entries(root: Path) -> list[str]:
    """Return the corpus entries that are absent under ``root``."""
    missing = []
    if not any(root.glob(str(_CFG["audio_glob"]))):
        missing.append(str(root / str(_CFG["audio_glob"])))
    if not (root / str(_CFG["labels_csv"])).is_file():
        missing.append(str(root / str(_CFG["labels_csv"])))
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding both the audio and the label table.

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
        "ANAD source tree not found. Checked:\n  "
        + "\n  ".join(checked)
        + "\n\nThe Kaggle download needs a login, so this recipe cannot fetch it. "
        f"Download\n  {_CFG['request_url']}\n"
        "extract the `*.wav` and `*.csv` members, and set "
        f"${_CFG['source_env_var']} to a directory holding:\n"
        f"  {_CFG['audio_glob']}\n"
        f"  {_CFG['labels_csv']}"
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory holding the manifests, token list and WAV files.

    Returns:
        Path: ``$ANAD_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / str(_CFG["data_path"])


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Join the label table with the wav chunks that exist.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``group`` (call and
        turn) and ``label``.

    Raises:
        ValueError: If a name or emotion is unexpected, or a name repeats.
        FileNotFoundError: If more than ``max_missing_audio`` chunks named in
            the table have no wav file.
    """
    excluded = {str(n) for n in _CFG["exclude_files"]}
    files: dict[str, Path] = {}
    for wav in sorted(source_root.glob(str(_CFG["audio_glob"]))):
        if wav.name in files:
            raise ValueError(f"Duplicate recording name: {wav} and {files[wav.name]}")
        files[wav.name] = wav

    entries = []
    missing = []
    with (source_root / str(_CFG["labels_csv"])).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            # The table writes names as `V2_1 (1).wav'` with a stray quote.
            name = row["name"].strip().rstrip("'")
            emotion = row["Emotion "].strip()
            fields = NAME_RE.match(name)
            if fields is None:
                raise ValueError(f"Unexpected recording name {name!r}")
            if emotion not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {emotion!r} for {name}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            call, turn, chunk = fields["call"], fields["turn"], fields["chunk"]
            # Some names use `V4-10` and others `V4_10`; the list uses `_`.
            canonical = f"V{call}_{turn} ({chunk}).wav"
            if canonical in excluded:
                excluded.discard(canonical)
                continue
            if name not in files:
                missing.append(name)
                continue
            entries.append(
                {
                    "utt_id": f"anad-V{call}_{turn}_{chunk}",
                    "source": str(files[name]),
                    "group": f"V{call}_{turn}",
                    "label": _EMOTION_MAP[emotion],
                }
            )

    if excluded:
        raise ValueError(f"`exclude_files` names not found: {sorted(excluded)[:5]}")
    allowed = int(_CFG["max_missing_audio"])
    if len(missing) > allowed:
        raise FileNotFoundError(
            f"{len(missing)} recording(s) named in the table are absent, starting "
            f"with {missing[0]}. `max_missing_audio` allows {allowed}, so this "
            "copy looks incomplete."
        )
    if missing:
        logger.warning("Skipped %d chunk(s) the release does not ship", len(missing))
    n_unlabeled = len(files) - len(entries) - len(_CFG["exclude_files"])
    if n_unlabeled:
        logger.warning(
            "Ignored %d wav chunk(s) with no label in the table", n_unlabeled
        )
    if len({e["utt_id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate utterance ids after name normalisation")
    if not entries:
        raise FileNotFoundError(f"No ANAD audio under {source_root}")
    logger.info("Read %d labelled chunk(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Assign whole call/turn groups to splits, label by label.

    Within each label the groups are taken largest first (ties broken with
    ``split_seed``), each going to the split furthest below its target count.

    Returns:
        dict[str, str]: Utterance id to split name.
    """
    ratios = {
        "train": float(_CFG["train_ratio"]),
        "valid": float(_CFG["valid_ratio"]),
        "test": float(_CFG["test_ratio"]),
    }
    rng = Random(int(_CFG["split_seed"]))
    by_label: dict[str, dict[str, list[str]]] = {}
    for e in entries:
        by_label.setdefault(e["label"], {}).setdefault(e["group"], []).append(
            e["utt_id"]
        )

    assignment: dict[str, str] = {}
    for label in sorted(by_label):
        groups = by_label[label]
        total = sum(len(v) for v in groups.values())
        counts = dict.fromkeys(ratios, 0)
        order = sorted(groups)
        rng.shuffle(order)
        order.sort(key=lambda g: -len(groups[g]))
        for group in order:
            split = max(ratios, key=lambda s: ratios[s] * total - counts[s])
            counts[split] += len(groups[group])
            for utt_id in groups[group]:
                assignment[utt_id] = split
    return assignment


class AnadBuilder(DatasetBuilder):
    """Prepare ANAD for the classification recipe.

    `build()` resamples every chunk to 16 kHz mono and writes one manifest per
    split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete source tree is reachable."""
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
        """Check that the corpus has been placed by hand.

        Raises:
            FileNotFoundError: With the download instructions if it is absent.
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
        """Resample the chunks and write one TSV manifest per split.

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
