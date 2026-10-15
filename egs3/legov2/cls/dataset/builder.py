"""LEGOv2 dataset builder (university archive, call-grouped split)."""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# Column positions in `interactions.csv` (60 columns per row).
N_COLUMNS = 60
CALL_ID, AUDIO_FILE, EMOTIONAL_STATE = 0, 54, 55
NULL = "N"  # MySQL `\N` once the escape character is removed.


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}
_DROP_STATES = {str(s) for s in _CFG["drop_states"]}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the LEGOv2 tree, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate directories that contain ``LEGOv2/``.
    """
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def read_rows(source_root: Path) -> list[dict[str, str]]:
    """Read the labelled exchanges of ``interactions.csv``.

    The file is Latin-1 and MySQL-escaped (backslash-escaped quotes inside
    quoted fields, a backslash-N for null).

    Args:
        source_root: Directory that contains ``LEGOv2/``.

    Returns:
        list[dict[str, str]]: Keys ``call``, ``audio`` and ``state`` for every
        exchange that has an emotion annotation.

    Raises:
        ValueError: If a row does not have the expected number of columns.
    """
    path = source_root / str(_CFG["interactions_csv"])
    rows = []
    with path.open("r", encoding="latin-1", newline="") as fh:
        for row in csv.reader(fh, delimiter=";", quotechar='"', escapechar="\\"):
            if len(row) != N_COLUMNS:
                raise ValueError(
                    f"{path}: expected {N_COLUMNS} columns, got {len(row)}"
                )
            if row[EMOTIONAL_STATE] in ("", NULL) or row[AUDIO_FILE] in ("", NULL):
                continue
            rows.append(
                {
                    "call": row[CALL_ID],
                    "audio": row[AUDIO_FILE],
                    "state": row[EMOTIONAL_STATE],
                }
            )
    return rows


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every labelled recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory that contains ``LEGOv2/``.

    Raises:
        FileNotFoundError: If no candidate holds all labelled recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (candidate / str(_CFG["interactions_csv"])).is_file():
            rows = read_rows(candidate)
            audio_root = candidate / str(_CFG["audio_subdir"])
            if (
                len(rows) >= _CFG["expected_labelled"]
                and sum((audio_root / r["audio"]).is_file() for r in rows)
                >= _CFG["expected_labelled"]
            ):
                return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete LEGOv2 not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the extracted "
        "`LEGOv2/` directory, or let `prepare_source` download it from "
        f"{_CFG['archive_url']}."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$LEGOV2_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(archive: Path, destination: Path) -> None:
    """Extract the recordings and the files in ``extract_members``.

    Args:
        archive: ``.zip`` archive to read.
        destination: Directory to extract into.

    Raises:
        zipfile.BadZipFile: If the archive is invalid.
        ValueError: If a member would be written outside ``destination``.
    """
    logger.info("Extracting %s", archive.name)
    root = destination.resolve()
    wanted = set(_CFG["extract_members"])
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member not in wanted and not member.lower().endswith(".wav"):
                continue
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member}")
            zf.extract(member, destination)


def _download_source(destination: Path) -> None:
    """Fetch the archive (unless already there) and unpack what the recipe uses.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the SHA-256 of the archive does not match.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = str(_CFG["archive_name"])
    archive_path = destination / archive_name

    if not archive_path.is_file():
        part = archive_path.with_suffix(archive_path.suffix + ".part")
        download_url(str(_CFG["archive_url"]), part, logger=logger)
        part.rename(archive_path)
    digest = _sha256(archive_path)
    if digest != str(_CFG["archive_sha256"]):
        raise ValueError(f"SHA-256 mismatch for {archive_name}: {digest}")
    _extract(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the labelled utterances.

    Args:
        source_root: Directory that contains ``LEGOv2/``.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``call`` and
        ``label``. Utterances in a state of ``drop_states`` are left out.

    Raises:
        ValueError: If a state is unknown or an utterance occurs twice.
        FileNotFoundError: If a recording is missing or none is found.
    """
    audio_root = source_root / str(_CFG["audio_subdir"])
    entries = []
    seen = set()
    dropped: Counter = Counter()
    for row in read_rows(source_root):
        state = row["state"]
        if state in _DROP_STATES:
            dropped[state] += 1
            continue
        if state not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotional state {state!r}. Known: {sorted(_EMOTION_MAP)}"
            )
        wav = audio_root / row["audio"]
        if not wav.is_file():
            raise FileNotFoundError(f"Labelled recording missing: {wav}")
        utt_id = "legov2-" + row["audio"].removesuffix(".wav").replace("/", "-")
        if utt_id in seen:
            raise ValueError(f"Duplicate utterance id: {utt_id}")
        seen.add(utt_id)
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "call": row["call"],
                "label": _EMOTION_MAP[state],
            }
        )
    for state, count in dropped.items():
        logger.warning("Dropped %d utterance(s) in state %r", count, state)
    if not entries:
        raise FileNotFoundError(f"No LEGOv2 audio under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d call(s)",
        len(entries),
        len({e["call"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Assign whole calls to splits, balancing the label proportions.

    Calls are visited rarest-label first (ties broken by ``split_seed``); each
    goes to the split whose label counts end up closest to the
    ``train_ratio`` / ``valid_ratio`` / ``test_ratio`` targets.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.
    """
    ratios = {
        "train": float(_CFG["train_ratio"]),
        "valid": float(_CFG["valid_ratio"]),
        "test": float(_CFG["test_ratio"]),
    }
    label_total = Counter(e["label"] for e in entries)
    target = {
        s: {lab: r * n for lab, n in label_total.items()} for s, r in ratios.items()
    }

    calls: dict[str, Counter] = defaultdict(Counter)
    for e in entries:
        calls[e["call"]][e["label"]] += 1
    order = sorted(calls)
    Random(int(_CFG["split_seed"])).shuffle(order)
    order.sort(key=lambda c: -sum(n / label_total[lab] for lab, n in calls[c].items()))

    current = {s: Counter() for s in ratios}
    split_of = {}
    for call in order:

        def cost(split: str) -> float:
            # Squared relative shortfall from the target, summed over labels.
            return sum(
                (
                    (current[split][lab] + n - target[split][lab]) ** 2
                    - (current[split][lab] - target[split][lab]) ** 2
                )
                / target[split][lab] ** 2
                for lab, n in calls[call].items()
            )

        best = min(ratios, key=cost)
        split_of[call] = best
        current[best].update(calls[call])
    return {e["utt_id"]: split_of[e["call"]] for e in entries}


class Legov2Builder(DatasetBuilder):
    """Prepare LEGOv2 for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes
    one manifest per split, drawn as a call-grouped split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the LEGOv2 tree is reachable."""
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
            logger.info("LEGOv2 already present; skipping download")
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
