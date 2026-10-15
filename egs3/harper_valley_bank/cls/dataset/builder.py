"""Harper Valley Bank dataset builder (3 polarity classes, one segment per row)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_CLASSES = [str(c) for c in _CFG["polarity_classes"]]
_ROLES = ("agent", "caller")


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the corpus, in order.

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


def _list_sids(root: Path) -> list[str]:
    transcripts = root / str(_CFG["transcript_subdir"])
    return sorted(p.stem for p in transcripts.glob("*.json"))


def missing_source_entries(root: Path) -> list[str]:
    """Return the corpus entries that are absent or incomplete under ``root``."""
    sids = _list_sids(root)
    missing = []
    if len(sids) < int(_CFG["expected_calls"]):
        missing.append(f"{len(sids)}/{_CFG['expected_calls']} transcripts")
    audio = root / str(_CFG["audio_subdir"])
    n_wav = sum(1 for r in _ROLES for _ in (audio / r).glob("*.wav"))
    if n_wav < 2 * int(_CFG["expected_calls"]):
        missing.append(f"{audio} ({n_wav}/{2 * int(_CFG['expected_calls'])} wav)")
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every transcript and recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root containing ``data/``.

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
        "Complete Harper Valley Bank corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding `data/`, "
        "or let `prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$HARPER_VALLEY_BANK_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _download_source(destination: Path) -> None:
    """Download the repository tarball and extract it into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / str(_CFG["archive_name"])
    part = archive.with_suffix(archive.suffix + ".part")
    logger.info("Downloading %s", _CFG["archive_url"])
    urllib.request.urlretrieve(str(_CFG["archive_url"]), part)
    part.replace(archive)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            # Drop the leading `<repo>-<branch>/` directory.
            member.name = member.name.split("/", 1)[1] if "/" in member.name else ""
            if member.name:
                tar.extract(member, destination)
    if _CFG["remove_archive"]:
        archive.unlink()


def read_segments(source_root: Path) -> list[dict]:
    """Read every transcript segment into one row with a polarity label.

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict]: Keys ``utt_id``, ``group`` (call id), ``source`` (wav of the
        speaking side), ``start`` and ``duration`` in seconds, and ``label``.

    Raises:
        ValueError: If a segment carries an unexpected class set.
    """
    audio = source_root / str(_CFG["audio_subdir"])
    rows = []
    for sid in _list_sids(source_root):
        path = source_root / str(_CFG["transcript_subdir"]) / f"{sid}.json"
        segments = json.loads(path.read_text(encoding="utf-8"))
        for seg in segments:
            scores = seg["emotion"]
            if sorted(scores) != sorted(_CLASSES):
                raise ValueError(f"Unexpected classes {sorted(scores)} in {path}")
            role = seg["speaker_role"]
            # `start_ms` is the position in the per-side wav; `offset_ms` is not.
            rows.append(
                {
                    "utt_id": f"hvb-{sid}-{int(seg['index']):04d}-{role}",
                    "group": sid,
                    "source": str(audio / role / f"{sid}.wav"),
                    "start": seg["start_ms"] / 1000.0,
                    "duration": seg["duration_ms"] / 1000.0,
                    "label": max(scores, key=scores.get),
                }
            )
    logger.info(
        "Read %d segment(s) from %d call(s)", len(rows), len({r["group"] for r in rows})
    )
    return rows


def assign_splits(rows: list[dict]) -> dict[str, str]:
    """Split whole calls at random so no call spans two splits.

    Calls are shuffled with ``split_seed`` and filled into test, then valid,
    then train until each holds about its ratio of the segments. Labels are not
    balanced across splits.

    Args:
        rows: Rows from :func:`read_segments`.

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
    for row in sorted(rows, key=lambda r: r["utt_id"]):
        by_group.setdefault(row["group"], []).append(row["utt_id"])
    groups = sorted(by_group)
    Random(int(_CFG["split_seed"])).shuffle(groups)

    total = len(rows)
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


def _cut(ffmpeg: str, row: dict, wav: Path) -> bool:
    """Cut one segment to 16 kHz mono; return False if it lies outside the audio."""
    info = sf.info(row["source"])
    if row["start"] >= info.duration:
        return False
    duration = min(row["duration"], info.duration - row["start"])
    part = wav.with_suffix(wav.suffix + ".part")
    subprocess.run(
        [
            ffmpeg,
            "-ss",
            f"{row['start']:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            row["source"],
            "-ac",
            "1",
            "-ar",
            str(_CFG["sampling_rate"]),
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
    return True


class HarperValleyBankBuilder(DatasetBuilder):
    """Prepare Harper Valley Bank for the classification recipe.

    `build()` cuts every transcript segment out of its call recording, resamples
    it to 16 kHz mono and writes one manifest per split, drawn as a random split
    of whole calls (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete corpus tree is reachable."""
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
        """Download and extract the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("Harper Valley Bank already present; skipping download")
            return
        _download_source(next(iter_source_candidates(recipe_root, source_dir)))
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
        """Cut the segments and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a conversion fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required to cut and resample segments")

        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)

        rows = read_segments(source_root)
        assignment = assign_splits(rows)

        def convert(row: dict) -> bool:
            wav = wav_dir / f"{row['utt_id']}.wav"
            return wav.is_file() or _cut(ffmpeg, row, wav)

        with ThreadPoolExecutor(max_workers=int(_CFG["convert_workers"])) as pool:
            kept = list(pool.map(convert, rows))
        n_outside = kept.count(False)
        if n_outside:
            logger.info("Dropped %d segment(s) starting after the audio end", n_outside)

        out: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for row, ok in zip(rows, kept):
            if ok:
                wav = wav_dir / f"{row['utt_id']}.wav"
                out[assignment[row["utt_id"]]].append(
                    (row["utt_id"], str(wav.resolve()), row["label"])
                )

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(out[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(out[split]), manifest)
