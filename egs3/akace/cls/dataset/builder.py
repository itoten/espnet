"""AkaCE (Akan Cinematic Emotions) dataset builder."""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
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
_SPLIT_MAP = {str(k): str(v) for k, v in _CFG["split_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the AkaCE files, in order.

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
    """Return the first candidate holding the label table and the segments.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding the label table and the segment directory.

    Raises:
        FileNotFoundError: If no candidate holds the table and the segment
            directory.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (candidate / _CFG["label_table"]).is_file() and (
            candidate / _CFG["segment_dir"]
        ).is_dir():
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "AkaCE files not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "`excel_files` and `audio/movie_audio_segments_wav` folders of the "
        "authors' Google Drive folder (see the readme)."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$AKACE_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _drop_duplicates(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Leave out the utterances listed in ``duplicate_ids`` of the config.

    Applies only when ``exclude_duplicates`` is true. Each listed utterance has
    the same segment and label as one that stays.

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


def _segment_stem(movie: str, start_s: float, end_s: float) -> str:
    start = round(start_s * 1000) + int(_CFG["start_offset_ms"])
    end = round(end_s * 1000) + int(_CFG["end_offset_ms"])
    return f"{movie}_{start}_{end}"


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Match the label table rows to the segment files.

    Args:
        source_root: Directory returned by :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``movie``, ``split``
        and ``label``. Rows without a segment file are left out and counted in
        the log. A second row with the same segment and label gets the suffix
        ``-r2`` (``-r3``, ...) so that ``duplicate_ids`` can name it.

    Raises:
        ValueError: If a label or split is unknown, the table has fewer than
            ``expected_rows`` rows, or an id of ``conflicting_ids`` is absent.
        FileNotFoundError: If no segment is found.
    """
    table = source_root / _CFG["label_table"]
    segment_dir = source_root / _CFG["segment_dir"]
    with table.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < int(_CFG["expected_rows"]):
        raise ValueError(
            f"{table} has {len(rows)} rows, expected {_CFG['expected_rows']}"
        )

    entries = []
    seen: dict[tuple[str, str], int] = {}
    no_segment = 0
    for row in rows:
        emotion = row["Emotion"].strip()
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {emotion!r}. Known: {sorted(_EMOTION_MAP)}"
            )
        if row["split"] not in _SPLIT_MAP:
            raise ValueError(f"Unknown split {row['split']!r}")
        label = _EMOTION_MAP[emotion]
        movie = row["Movie Title"]
        stem = _segment_stem(
            movie, float(row["start_seconds"]), float(row["end_seconds"])
        )
        wav = segment_dir / movie / f"{stem}.wav"
        if not wav.is_file():
            no_segment += 1
            continue
        key = (stem, label)
        seen[key] = seen.get(key, 0) + 1
        suffix = "" if seen[key] == 1 else f"-r{seen[key]}"
        entries.append(
            {
                "utt_id": f"akace-{stem}-{label}{suffix}",
                "source": str(wav),
                "movie": movie,
                "split": _SPLIT_MAP[row["split"]],
                "label": label,
            }
        )
    if no_segment:
        logger.warning("%d table row(s) have no segment file; left out", no_segment)
    if not entries:
        raise FileNotFoundError(f"No AkaCE segment under {segment_dir}")
    logger.info(
        "Read %d utterance(s) from %d movie(s)",
        len(entries),
        len({e["movie"] for e in entries}),
    )
    conflicting = {str(i) for i in _CFG["conflicting_ids"]}
    missing = conflicting - {e["utt_id"] for e in entries}
    if missing:
        raise ValueError(f"`conflicting_ids` not found: {sorted(missing)}")
    entries = [e for e in entries if e["utt_id"] not in conflicting]
    return _drop_duplicates(entries)


class AkaceBuilder(DatasetBuilder):
    """Prepare AkaCE for the classification recipe.

    The segments are cut by the authors from the movie audio, so `build()`
    converts each to 16 kHz mono and writes one manifest per split, following
    the ``split`` column of the label table.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the AkaCE files are reachable."""
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
        """Check that the files, which are fetched by hand, are in place.

        Raises:
            FileNotFoundError: If the files are not found.
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
        """Resample the segments and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location of the AkaCE files.

        Raises:
            FileNotFoundError: If the files or ``ffmpeg`` cannot be found.
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

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        converted: dict[str, Path] = {}
        for entry in read_entries(source_root):
            # Rows sharing a segment and label use one converted file.
            wav = converted.get(entry["source"])
            if wav is None:
                wav = wav_dir / f"{Path(entry['source']).stem}.wav"
                converted[entry["source"]] = wav
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
