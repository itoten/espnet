"""eNTERFACE'05 dataset builder."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import zipfile
from importlib import resources
from pathlib import Path
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

SUBJECT_RE = re.compile(r"^subject\s*(?P<number>\d+)$", re.I)
SENTENCE_RE = re.compile(r"^sentence\s*(?P<number>\d+)$", re.I)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k).lower(): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the corpus tree, in order.

    Each base is tried both directly and with the archive's own subdirectory
    appended, so an archive unpacked as-is and a flattened copy both resolve.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate corpus roots, most specific first.
    """
    bases = []
    if source_dir:
        bases.append(Path(source_dir).expanduser())
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        bases.append(Path(env_value).expanduser())
    bases.append(recipe_root / _CFG["dataset_path"])

    for base in bases:
        yield base
        yield base / str(_CFG["archive_subdir"])


def iter_subject_dirs(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(speaker id, directory)`` for every subject folder under ``root``.

    Args:
        root: Candidate corpus root.

    Yields:
        tuple[str, Path]: Numeric speaker id and its directory.
    """
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        match = SUBJECT_RE.match(child.name) if child.is_dir() else None
        if match:
            yield match["number"], child


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding subject folders with recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding the ``subject <n>`` folders.

    Raises:
        FileNotFoundError: If no candidate holds any.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        for _, subject_dir in iter_subject_dirs(candidate):
            if any(subject_dir.rglob("*.avi")):
                return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "eNTERFACE'05 subject folders not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$ENTERFACE05_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_zip(archive: Path, destination: Path) -> None:
    """Extract one zip archive into ``destination``.

    Members are checked against the destination first: ``ZipFile.extractall``
    does not reject absolute paths or ``..`` components on its own.

    Args:
        archive: Archive to read.
        destination: Directory to extract into.

    Raises:
        ValueError: If a member would be written outside ``destination``.
        zipfile.BadZipFile: If the archive is invalid.
    """
    logger.info("Extracting %s", archive.name)
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError(f"{archive.name} escapes its destination: {name}")
        zf.extractall(path=destination)


def find_archive(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path | None:
    """Return a hand-placed archive in any of the candidate directories.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path | None: The archive, or None when none is there.
    """
    archive_name = str(_CFG["archive_name"])
    for candidate in iter_source_candidates(recipe_root, source_dir):
        archive = candidate / archive_name
        if archive.is_file():
            return archive
    return None


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking speaker and label from the directories.

    The file names cannot be trusted. Twenty-three carry a typo, and every
    recording under ``subject 11`` is named ``s12_*`` even though its contents
    differ from the ones under ``subject 12`` -- reading the speaker off the
    name would put one speaker in two splits. The directory names are regular,
    so they are used instead.

    Args:
        source_root: Directory holding the ``subject <n>`` folders.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If an emotion directory is not in ``emotion_map``, or if
            two recordings would share an utterance id.
        FileNotFoundError: If no recording is found.
    """
    excluded = {str(x) for x in _CFG.get("exclude_speakers", [])}
    entries = []
    for speaker, subject_dir in iter_subject_dirs(source_root):
        if speaker in excluded:
            logger.info("Skipping speaker %s; see dataset/config.yaml", speaker)
            continue
        for emotion_dir in sorted(d for d in subject_dir.iterdir() if d.is_dir()):
            recordings = sorted(emotion_dir.rglob("*.avi"))
            # One subject ships an empty `neutral` directory, an emotion the
            # corpus does not otherwise have. Judge a directory by what is in
            # it, so an empty one cannot stop the build.
            if not recordings:
                continue
            emotion = emotion_dir.name.strip().lower()
            if emotion not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion directory {emotion_dir}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            for avi in recordings:
                # One subject stores its recordings directly under the emotion
                # directory, without the `sentence <m>` level.
                match = SENTENCE_RE.match(avi.parent.name)
                sentence = match["number"] if match else "0"
                entries.append(
                    {
                        "utt_id": f"enterface05-s{speaker}-{emotion}-{sentence}",
                        "source": str(avi),
                        "speaker": speaker,
                        "label": _EMOTION_MAP[emotion],
                    }
                )

    seen: dict[str, str] = {}
    for entry in entries:
        clash = seen.get(entry["utt_id"])
        if clash is not None:
            raise ValueError(
                f"Two recordings share the id {entry['utt_id']}: "
                f"{clash} and {entry['source']}"
            )
        seen[entry["utt_id"]] = entry["source"]

    if not entries:
        raise FileNotFoundError(f"No eNTERFACE'05 audio under {source_root}")
    logger.info(
        "Read %d recording(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(speakers: list[str]) -> dict[str, str]:
    """Deal sorted speaker ids out to the configured splits.

    Args:
        speakers: Speaker id of every recording, repeats allowed.

    Returns:
        dict[str, str]: Speaker id to split name.

    Raises:
        ValueError: If the counts do not account for every speaker.
    """
    counts = {str(k): int(v) for k, v in _CFG["split_speaker_counts"].items()}
    ordered = sorted(set(speakers), key=int)
    if sum(counts.values()) != len(ordered):
        raise ValueError(
            f"`split_speaker_counts` sums to {sum(counts.values())} but the "
            f"corpus has {len(ordered)} speakers "
            f"({', '.join(f'{k}={v}' for k, v in counts.items())}). "
            "Adjust the counts in dataset/config.yaml so every speaker lands "
            "in exactly one split."
        )
    assignment = {}
    start = 0
    for split, count in counts.items():
        for speaker in ordered[start : start + count]:
            assignment[speaker] = split
        start += count
    return assignment


class ENTERFACE05Builder(DatasetBuilder):
    """Prepare eNTERFACE'05 for the classification recipe.

    The recordings are AVI files carrying 48 kHz stereo PCM, so `build()`
    extracts the audio as 16 kHz mono WAV and writes one manifest per split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a corpus tree with recordings is reachable."""
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
        """Unpack the hand-placed archive unless the tree is already there.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If neither the tree nor the archive is present.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("eNTERFACE'05 already present; skipping extraction")
            return

        archive = find_archive(recipe_root, source_dir)
        if archive is None:
            raise FileNotFoundError(
                f"{_CFG['archive_name']} was not found in any of "
                + ", ".join(
                    str(c) for c in iter_source_candidates(recipe_root, source_dir)
                )
                + ".\nThe corpus cannot be fetched from a script: its site sends "
                "an incomplete certificate chain, which browsers work around but "
                "ordinary clients reject. Download it from "
                "https://enterface.net/enterface05/emotion.html and put it in "
                f"${_CFG['source_env_var']}; this stage unpacks it from there."
            )
        _extract_zip(archive, archive.parent)
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
        """Extract the audio and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If an extraction fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required to extract the audio from the AVI files"
            )

        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)
        sampling_rate = str(_CFG["sampling_rate"])

        entries = read_entries(source_root)
        assignment = assign_splits([e["speaker"] for e in entries])

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            wav = wav_dir / f"{entry['utt_id']}.wav"
            if not wav.is_file():
                # `-f wav` sets the format, so the `.part` extension is fine.
                part = wav.with_suffix(wav.suffix + ".part")
                subprocess.run(
                    [
                        ffmpeg,
                        "-i",
                        entry["source"],
                        "-vn",
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
            rows[assignment[entry["speaker"]]].append(
                (entry["utt_id"], str(wav.resolve()), entry["label"])
            )

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
