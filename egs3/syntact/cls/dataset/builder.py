"""SyntAct dataset builder (Zenodo archive, official speaker-disjoint split)."""

from __future__ import annotations

import csv
import hashlib
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
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `synthesized_audio/de1_happy_p000000.wav`: voice, intended emotion, running number.
FILENAME_RE = re.compile(
    r"^(?P<speaker>de\d)_(?P<emotion>[a-z]+)_p(?P<number>\d+)\.wav$"
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
    """Yield the directories that may hold the SyntAct source tree, in order.

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
    """Return the first candidate holding the complete set of recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding the ``release_dir`` folder.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (candidate / _CFG["release_dir"] / "db.files.csv").is_file() and sum(
            1
            for wav in (candidate / _CFG["release_dir"]).rglob("*.wav")
            if FILENAME_RE.match(wav.name)
        ) >= _CFG["expected_files"]:
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete SyntAct release not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "extracted `syntact_cat/` folder, or let `prepare_source` download them "
        "from Zenodo."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$SYNTACT_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_wavs(archive: Path, destination: Path) -> None:
    """Extract the WAV and CSV files of the archive into ``destination``.

    Args:
        archive: ``.zip`` archive to read.
        destination: Directory to extract into.

    Raises:
        zipfile.BadZipFile: If the archive is invalid.
        ValueError: If a member would be written outside ``destination``.
    """
    logger.info("Extracting %s", archive.name)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not member.lower().endswith((".wav", ".csv")):
                continue
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member}")
            zf.extract(member, destination)


def _download_source(destination: Path) -> None:
    """Fetch the archive and unpack its WAV and CSV files.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the MD5 checksum does not match.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = str(_CFG["archive_name"])

    archive_path = destination / archive_name
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    download_url(str(_CFG["archive_url"]), part, logger=logger)
    md5 = hashlib.md5()
    with part.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            md5.update(chunk)
    if md5.hexdigest() != str(_CFG["archive_md5"]):
        raise ValueError(f"MD5 mismatch for {archive_name}: {md5.hexdigest()}")
    part.rename(archive_path)
    _extract_wavs(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def _md5(path: Path) -> str:
    """Return the MD5 hex digest of a file, read in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_no_duplicates(entries: list[dict[str, str]]) -> None:
    """Raise if two recordings are byte-identical."""
    owner: dict[str, str] = {}
    for entry in entries:
        digest = _md5(Path(entry["source"]))
        if digest in owner:
            raise ValueError(
                f"Identical file: {entry['utt_id']} and {owner[digest]}; "
                "add one of them to `exclude_files`"
            )
        owner[digest] = entry["utt_id"]


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings with the official split and desired emotion.

    Files named in ``exclude_files`` are skipped.

    Args:
        source_root: Directory holding the ``release_dir`` folder.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker``, ``split``
        (``train`` or ``test``) and ``label``.

    Raises:
        ValueError: If an emotion is unknown, a file name does not parse or
            disagrees with the voice table, a file is listed twice, an excluded
            name is not found, or two remaining recordings are byte-identical.
        FileNotFoundError: If a table or a recording is missing.
    """
    release = source_root / str(_CFG["release_dir"])
    with (release / "db.files.csv").open(encoding="utf-8") as fh:
        voice_of = {row["file"]: row["speaker"] for row in csv.DictReader(fh)}
    excluded = {str(n) for n in _CFG["exclude_files"]}
    entries = []
    seen: set[str] = set()
    for split, table in _CFG["tables"].items():
        with (release / str(table)).open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            name = row["file"]
            fields = FILENAME_RE.match(Path(name).name)
            if fields is None:
                raise ValueError(f"Unexpected file name: {name}")
            if voice_of.get(name) != fields["speaker"]:
                raise ValueError(f"Voice table and file name disagree: {name}")
            if row["emotion"] not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown emotion {row['emotion']!r} in {name}. "
                    f"Known: {sorted(_EMOTION_MAP)}"
                )
            if name in seen:
                raise ValueError(f"Recording listed twice: {name}")
            seen.add(name)
            if name in excluded:
                excluded.discard(name)
                continue
            wav = release / name
            if not wav.is_file():
                raise FileNotFoundError(f"Missing recording: {wav}")
            entries.append(
                {
                    "utt_id": f"syntact-{wav.stem}",
                    "source": str(wav),
                    "speaker": fields["speaker"],
                    "split": split,
                    "label": _EMOTION_MAP[row["emotion"]],
                }
            )
    if excluded:
        raise ValueError(f"`exclude_files` names not found: {sorted(excluded)}")
    if not entries:
        raise FileNotFoundError(f"No SyntAct audio under {source_root}")
    _check_no_duplicates(entries)
    logger.info(
        "Read %d utterance(s) from %d voice(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Keep the official train/test split and carve valid out of train by voice.

    The official split already separates the voices (``de1``-``de4`` train,
    ``de6``/``de7`` test). The ``valid_speakers`` voices move from train to valid.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If a ``valid_speakers`` voice is not a training voice.
    """
    valid_speakers = {str(s) for s in _CFG["valid_speakers"]}
    train_voices = {e["speaker"] for e in entries if e["split"] == "train"}
    if not valid_speakers < train_voices:
        raise ValueError(
            f"valid_speakers {sorted(valid_speakers)} must be a proper subset of "
            f"the training voices {sorted(train_voices)}"
        )
    return {
        e["utt_id"]: (
            "valid"
            if e["split"] == "train" and e["speaker"] in valid_speakers
            else e["split"]
        )
        for e in sorted(entries, key=lambda e: e["utt_id"])
    }


class SyntactBuilder(DatasetBuilder):
    """Prepare SyntAct for the classification recipe.

    `build()` copies every recording (already 16 kHz mono) into the shared
    layout and writes
    one manifest per split, drawn as the official voice-disjoint split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a SyntAct source tree is reachable."""
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
            logger.info("SyntAct already present; skipping download")
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
        """Convert the recordings and write one TSV manifest per split.

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
