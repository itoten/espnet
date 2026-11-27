"""EARS dataset builder (the emotional recordings only)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `emo_adoration_freeform.wav` -> emotion "adoration", task "freeform".
# `emo_<emotion>_sentences.wav`: three sentences read in that tone.
# `emo_<emotion>_freeform.wav`: an image described in that tone.
EMOTION_FILE_RE: re.Pattern


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
EMOTION_FILE_RE = re.compile(str(_CFG["emotion_file_re"]))
_LABEL_FIX = {str(k): str(v) for k, v in _CFG["label_fix"].items()}
_SPEAKERS = [str(s) for s in _CFG["speakers"]]


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the EARS source, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate source roots, most specific first.
    """
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def _speaker_location(root: Path, speaker: str) -> Path | None:
    """Return the extracted folder or the archive of one speaker, if present."""
    folder = root / speaker
    if folder.is_dir() and any(folder.glob("emo_*.wav")):
        return folder
    for archive in (
        root / str(_CFG["archive_subdir"]) / f"{speaker}.zip",
        root / f"{speaker}.zip",
    ):
        if archive.is_file():
            return archive
    return None


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate that holds at least one speaker.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Source root with extracted ``pNNN/`` folders or ``pNNN.zip``
        archives (directly or under ``ARCHIVES/``).

    Raises:
        FileNotFoundError: If no candidate holds any speaker.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir() and any(
            _speaker_location(candidate, s) is not None for s in _SPEAKERS
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "EARS audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding `pNNN/` "
        "folders or `pNNN.zip` archives, or let `prepare_source` download them "
        "from the GitHub release."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EARS_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _download_speaker(speaker: str, destination: Path) -> None:
    """Fetch one speaker's archive, keep its ``emo_*`` recordings, drop the rest.

    Args:
        speaker: Speaker code such as ``p001``.
        destination: Source root; the recordings go to ``destination/<speaker>/``.

    Raises:
        urllib.error.URLError: If the download fails.
        zipfile.BadZipFile: If the archive is invalid.
        ValueError: If a member would be written outside its folder.
    """
    destination.mkdir(parents=True, exist_ok=True)
    url = str(_CFG["base_url"]).rstrip("/") + f"/{speaker}.zip"
    archive = destination / f"{speaker}.zip"
    part = archive.with_suffix(".zip.part")
    download_url(url, part, logger=logger)
    part.rename(archive)
    folder = (destination / speaker).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename.split("/")[-1]
            if EMOTION_FILE_RE.match(name):
                target = (folder / name).resolve()
                if not target.is_relative_to(folder):
                    raise ValueError(f"{archive.name} escapes its folder: {info.filename}")
                target.write_bytes(zf.read(info))
    archive.unlink()


def _download_source(destination: Path) -> None:
    """Download every speaker that is not extracted yet (66 GB of transfer)."""
    for speaker in _SPEAKERS:
        if _speaker_location(destination, speaker) is None:
            logger.info("Downloading %s", speaker)
            _download_speaker(speaker, destination)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the emotional recordings, taking the label from the file name.

    Args:
        source_root: Source root (see :func:`resolve_source_root`).

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` (a wav path, or
        ``<archive>!<member>``), ``speaker``, ``task`` and ``label``.

    Raises:
        FileNotFoundError: If no emotional recording is found.
    """
    entries = []
    missing = []
    for speaker in _SPEAKERS:
        location = _speaker_location(source_root, speaker)
        if location is None:
            missing.append(speaker)
            continue
        if location.is_dir():
            members = [(p.name, str(p)) for p in sorted(location.glob("emo_*.wav"))]
        else:
            with zipfile.ZipFile(location) as zf:
                members = [
                    (i.filename.split("/")[-1], f"{location}!{i.filename}")
                    for i in zf.infolist()
                ]
        for name, source in sorted(members):
            fields = EMOTION_FILE_RE.match(name)
            if fields is None:
                continue
            emotion = _LABEL_FIX.get(fields["emotion"], fields["emotion"])
            entries.append(
                {
                    "utt_id": f"ears-{speaker}-{emotion}-{fields['task']}",
                    "source": source,
                    "speaker": speaker,
                    "task": fields["task"],
                    "label": emotion,
                }
            )
    if missing:
        logger.warning("%d speaker(s) not found: %s", len(missing), ", ".join(missing))
    if not entries:
        raise FileNotFoundError(f"No EARS emotional recording under {source_root}")
    logger.info(
        "Read %d recording(s) from %d speaker(s), %d label(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
        len({e["label"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Assign each recording to the split of its speaker.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If a speaker is in no split's range.
    """
    ranges = {
        split: (str(first), str(last))
        for split, (first, last) in _CFG["split_speakers"].items()
    }
    assignment: dict[str, str] = {}
    for entry in entries:
        for split, (first, last) in ranges.items():
            if first <= entry["speaker"] <= last:
                assignment[entry["utt_id"]] = split
                break
        else:
            raise ValueError(f"Speaker {entry['speaker']} is in no split range")
    return assignment


def _convert(ffmpeg: str, source: str, wav: Path, sampling_rate: str) -> None:
    """Write one recording as 16 kHz mono 16-bit PCM, reading it in place."""
    part = wav.with_suffix(wav.suffix + ".part")
    command = [
        ffmpeg, "-i", "pipe:0", "-ac", "1", "-ar", sampling_rate,
        "-c:a", "pcm_s16le", "-f", "wav", "-y", "-hide_banner", "-loglevel", "error",
        str(part),
    ]
    if "!" in source:
        archive, member = source.split("!", 1)
        with zipfile.ZipFile(archive) as zf:
            data = zf.read(member)
    else:
        data = Path(source).read_bytes()
    subprocess.run(command, input=data, check=True)
    part.replace(wav)


class EARSBuilder(DatasetBuilder):
    """Prepare the EARS emotional recordings for the classification recipe.

    The recordings are 48 kHz mono 32-bit float WAV, so `build()` converts every
    one to 16 kHz mono 16-bit PCM and writes one manifest per split, by the
    corpus paper's speaker partition (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether an EARS source is reachable."""
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
        """Download the speakers that are not available yet.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location, tried before the environment variable
                and the recipe-local download directory.

        Raises:
            FileNotFoundError: If nothing is available afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        candidates = list(iter_source_candidates(recipe_root, source_dir))
        for candidate in candidates:
            if candidate.is_dir() and all(
                _speaker_location(candidate, s) is not None for s in _SPEAKERS
            ):
                logger.info("EARS already present in %s; skipping download", candidate)
                return
        _download_source(candidates[0])
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
        num_workers: int = 8,
        **_kwargs,
    ) -> None:
        """Convert the recordings and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location.
            num_workers: ffmpeg processes run at the same time.

        Raises:
            FileNotFoundError: If the source or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a conversion fails.
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

        todo = [
            (e["source"], wav_dir / f"{e['utt_id']}.wav")
            for e in entries
            if not (wav_dir / f"{e['utt_id']}.wav").is_file()
        ]
        logger.info("Converting %d of %d recording(s)", len(todo), len(entries))
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            list(pool.map(lambda t: _convert(ffmpeg, t[0], t[1], sampling_rate), todo))

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            wav = wav_dir / f"{entry['utt_id']}.wav"
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
