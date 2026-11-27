"""VESUS dataset builder (actors reading one script in six instructed emotions)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
_AUDIO_RE = re.compile(str(_CFG["audio_re"]))
_LABELS = {str(k): str(v) for k, v in _CFG["label_folders"].items()}
_SPLIT_OF_SPEAKER = {
    str(speaker): str(split)
    for split, speakers in _CFG["split_speakers"].items()
    for speaker in speakers
}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the VESUS release, in order.

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


def list_members(root: Path) -> list[tuple[str, str]]:
    """List the release's audio files as ``(path inside the release, source)``.

    ``source`` is a file path when the release is unpacked under ``root``, or
    ``<archive>!<member>`` when ``root`` holds the archive ``archive_name``,
    which is then read in place. Paths that do not match ``audio_re`` (macOS
    resource forks, ``.DS_Store``, the other folders of the release) are not
    listed.
    """
    archive = root / str(_CFG["archive_name"])
    if archive.is_file():
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        return sorted((n, f"{archive}!{n}") for n in names if _AUDIO_RE.match(n))
    audio = root / "Audio"
    if not audio.is_dir():
        return []
    found = []
    for path in audio.rglob("*.wav"):
        rel = path.relative_to(root).as_posix()
        if _AUDIO_RE.match(rel):
            found.append((rel, str(path)))
    return sorted(found)


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the expected number of recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding ``archive_name`` or the unpacked ``Audio/``.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    expected = int(_CFG["expected_recordings"])
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        found = len(list_members(candidate))
        if found == expected:
            return candidate
        checked.append(f"{candidate} ({found} of {expected} recordings)")
    raise FileNotFoundError(
        "VESUS not found. Checked:\n  "
        + "\n  ".join(checked)
        + "\nVESUS is distributed by the NSA lab at Johns Hopkins University after "
        "a request form (see readme.md, Download); it cannot be downloaded by a "
        f"script. Put `{_CFG['archive_name']}` (or its unpacked `Audio/`) in a "
        f"directory and set ${_CFG['source_env_var']} to it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$VESUS_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings; the folder is the label, the actor the speaker.

    Args:
        source_root: Source root (see :func:`resolve_source_root`).

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a folder is not in ``label_folders`` or an id repeats.
    """
    entries = []
    seen: set[str] = set()
    for rel, source in list_members(source_root):
        fields = _AUDIO_RE.match(rel)
        folder = fields["folder"]
        if folder not in _LABELS:
            raise ValueError(f"Unknown label folder {folder!r} in {rel}")
        utt_id = f"vesus-{fields['actor']}-{_LABELS[folder]}-{fields['number']}"
        if utt_id in seen:
            raise ValueError(f"Duplicate utterance id: {utt_id}")
        seen.add(utt_id)
        entries.append(
            {
                "utt_id": utt_id,
                "source": source,
                "speaker": fields["actor"],
                "label": _LABELS[folder],
            }
        )
    logger.info(
        "Read %d recording(s) from %d actor(s); per label: %s",
        len(entries),
        len({e["speaker"] for e in entries}),
        dict(sorted(Counter(e["label"] for e in entries).items())),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Assign each recording to the split of its actor (``split_speakers``).

    Raises:
        ValueError: If an actor is in no split.
    """
    assignment = {}
    for entry in entries:
        split = _SPLIT_OF_SPEAKER.get(entry["speaker"])
        if split is None:
            raise ValueError(f"Actor {entry['speaker']} is in no split")
        assignment[entry["utt_id"]] = split
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


class VESUSBuilder(DatasetBuilder):
    """Prepare VESUS for the classification recipe.

    The release cannot be downloaded by a script; `prepare_source` only checks
    that it is in place. `build()` reads the archive in place (or the unpacked
    ``Audio/`` folder), converts every recording to 16 kHz mono and writes one
    manifest per split, by actor (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete VESUS release is reachable."""
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
        """Check the release is in place; it cannot be downloaded by a script.

        Raises:
            FileNotFoundError: If the release is not found (the message says
                how to obtain it).
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
        num_workers: int = 8,
        **_kwargs,
    ) -> None:
        """Convert the recordings and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location.
            num_workers: ffmpeg processes run at the same time.

        Raises:
            FileNotFoundError: If the release or ``ffmpeg`` cannot be found.
            ValueError: If a label folder is unknown or an actor is in no split.
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
            counts = Counter(label for _, _, label in rows[split])
            logger.info(
                "%s: wrote %d entries to %s (%s)",
                split, len(rows[split]), manifest,
                ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
            )
