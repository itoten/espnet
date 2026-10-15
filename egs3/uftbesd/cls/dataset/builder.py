"""UFTBESD dataset builder (acted Bangla speech, 7 emotions)."""

from __future__ import annotations

import csv
import hashlib
import urllib.request
import logging
import os
import shutil
import subprocess
import zipfile
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
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the UFTBESD source tree, in order.

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
    """Return the first candidate holding the label table and every recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding the metadata CSV and the wav files.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and (candidate / str(_CFG["labels_csv"])).is_file()
            and sum(1 for _ in candidate.rglob("*.wav")) >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete UFTBESD corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['labels_csv']}` and the extracted wav files, or let "
        "`prepare_source` download them from Mendeley Data."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$UFTBESD_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _fetch(url: str, path: Path) -> None:
    """Download ``url`` to ``path``; Mendeley rejects the default user agent."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as resp, path.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _md5(path: Path) -> str:
    """Return the MD5 hex digest of a file, read in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_source(destination: Path) -> None:
    """Fetch the metadata CSV and the archive, then unpack the wav files.

    Args:
        destination: Directory to download into and unpack under.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a checksum does not match.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for item in _CFG["files"]:
        name = str(item["name"])
        target = destination / name
        part = target.with_suffix(target.suffix + ".part")
        _fetch(str(item["url"]), part)
        digest = _sha256(part)
        if digest != str(item["sha256"]):
            raise ValueError(f"SHA-256 mismatch for {name}: {digest}")
        part.rename(target)
    archive_path = destination / str(_CFG["archive_name"])
    logger.info("Extracting %s", archive_path.name)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(destination / "extracted")
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the metadata table into one row per recording.

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If an emotion is not in ``emotion_map`` or a file name
            appears twice.
        FileNotFoundError: If a recording named in the table is absent.
    """
    wavs = {p.name: p for p in source_root.rglob("*.wav")}
    entries = []
    seen: set[str] = set()
    with (source_root / str(_CFG["labels_csv"])).open(
        "r", encoding="utf-8-sig", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            name = row["file_name"].strip()
            emotion = row["emotion_label"].strip()
            if emotion not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {emotion!r} for {name}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            if name in seen:
                raise ValueError(f"Duplicate recording name: {name}")
            seen.add(name)
            if name not in wavs:
                raise FileNotFoundError(
                    f"Recording named in the table is absent: {name}"
                )
            entries.append(
                {
                    "utt_id": f"uftbesd-{Path(name).stem}",
                    "source": str(wavs[name]),
                    "speaker": row["speaker_id"].strip(),
                    "label": _EMOTION_MAP[emotion],
                }
            )
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def _group_speakers(entries: list[dict[str, str]]) -> dict[str, str]:
    """Map each speaker to a group id; speakers sharing identical audio merge.

    Some recordings are byte-identical across speakers, so such speakers must
    land in the same split to keep the test set free of copies of training audio.
    """
    parent: dict[str, str] = {e["speaker"]: e["speaker"] for e in entries}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    for entry in entries:
        digest = _md5(Path(entry["source"]))
        other = owner.setdefault(digest, entry["speaker"])
        parent[find(entry["speaker"])] = find(other)
    return {speaker: find(speaker) for speaker in parent}


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances so that no speaker appears in two splits.

    Speakers whose recordings share identical audio are merged into one group
    (see :func:`_group_speakers`). Groups are shuffled with ``split_seed`` and
    filled into test, valid and train in that order until each holds its
    ``test_ratio``/``valid_ratio`` share of the utterances.

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

    group_of = _group_speakers(entries)
    members: dict[str, list[str]] = {}
    for entry in entries:
        members.setdefault(group_of[entry["speaker"]], []).append(entry["utt_id"])
    groups = sorted(members)
    Random(int(_CFG["split_seed"])).shuffle(groups)

    targets = [("test", test_ratio), ("valid", valid_ratio)]
    assignment: dict[str, str] = {}
    cursor = 0
    for split, ratio in targets:
        goal = ratio * len(entries)
        count = 0
        while cursor < len(groups) and count < goal:
            for utt_id in members[groups[cursor]]:
                assignment[utt_id] = split
            count += len(members[groups[cursor]])
            cursor += 1
    for group in groups[cursor:]:
        for utt_id in members[group]:
            assignment[utt_id] = "train"
    return assignment


class UftbesdBuilder(DatasetBuilder):
    """Prepare UFTBESD for the classification recipe.

    `build()` resamples every recording from 44.1 kHz to 16 kHz mono and writes
    one manifest per split, with speakers kept disjoint (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a UFTBESD source tree is reachable."""
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
            logger.info("UFTBESD already present; skipping download")
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
