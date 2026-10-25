"""UTeMo dataset builder."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import zipfile
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `Happiness_Actor10_P2_F3.avi`: emotion, actor 10, repetition 2, phrase 3.
FILENAME_RE = re.compile(
    r"^(?P<emotion>[A-Za-z]+)_Actor(?P<speaker>\d+)_P(?P<rep>\d+)_"
    r"F(?P<phrase>\d+)\.avi$"
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
    """Yield the directories that may hold the UTeMo source tree, in order.

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
    """Return the first candidate holding every UTeMo recording.

    Recordings are searched recursively, since they sit in a single `Utemo/`
    directory.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding all ``<emotion>_Actor<n>_P<m>_F<l>.avi``
        files below it.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and sum(
                1 for avi in candidate.rglob("*.avi") if FILENAME_RE.match(avi.name)
            )
            >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete UTeMo videos not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "extracted `Utemo/` directory, or let `prepare_source` download it from "
        "Mendeley Data."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$UTEMO_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_videos(archive: Path, destination: Path) -> None:
    """Extract the AVI files of the archive into ``destination``.

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
            if not member.lower().endswith(".avi"):
                continue
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member}")
            zf.extract(member, destination)


def _download_source(destination: Path) -> None:
    """Fetch the archive, check its SHA-256 and unpack the videos.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the archive has the wrong SHA-256.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / str(_CFG["archive_name"])
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    download_url(str(_CFG["archive_url"]), part, logger=logger)
    digest = _sha256(part)
    if digest != str(_CFG["archive_sha256"]):
        raise ValueError(f"SHA-256 mismatch for {archive_path.name}: {digest}")
    part.rename(archive_path)
    _extract_videos(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking emotion and actor from the file name.

    Args:
        source_root: Corpus root with the videos somewhere below it.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a file's emotion is unknown, two files share a name
            (the name is the utterance id), or an `exclude_files` name is absent.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: dict[str, Path] = {}
    skipped = 0
    excluded = set(_CFG["exclude_files"])
    for avi in sorted(source_root.rglob("*.avi")):
        if avi.name in excluded:
            excluded.discard(avi.name)
            continue
        fields = FILENAME_RE.match(avi.name)
        if fields is None:
            skipped += 1
            continue
        emotion = fields["emotion"].lower()
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {fields['emotion']!r} in {avi}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        if avi.stem in seen:
            raise ValueError(f"Duplicate recording name: {avi} and {seen[avi.stem]}")
        seen[avi.stem] = avi
        entries.append(
            {
                "utt_id": f"utemo-{avi.stem}",
                "source": str(avi),
                "speaker": fields["speaker"],
                "label": _EMOTION_MAP[emotion],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if excluded:
        raise ValueError(f"`exclude_files` names not found: {sorted(excluded)}")
    if not entries:
        raise FileNotFoundError(f"No UTeMo videos under {source_root}")
    logger.info(
        "Read %d recording(s) from %d actor(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole actors: ``n_test_actors``, ``n_valid_actors``, rest train.

    Actors are shuffled with ``split_seed``, so every split has actors the
    others never see.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If fewer actors exist than the two held-out splits need
            plus one for training.
    """
    n_test = int(_CFG["n_test_actors"])
    n_valid = int(_CFG["n_valid_actors"])
    actors = sorted({e["speaker"] for e in entries}, key=int)
    if len(actors) < n_test + n_valid + 1:
        raise ValueError(f"Only {len(actors)} actors for {n_test}+{n_valid}+train")
    Random(int(_CFG["split_seed"])).shuffle(actors)
    split_of = {a: "test" for a in actors[:n_test]}
    split_of.update({a: "valid" for a in actors[n_test : n_test + n_valid]})
    return {e["utt_id"]: split_of.get(e["speaker"], "train") for e in entries}


class UtemoBuilder(DatasetBuilder):
    """Prepare UTeMo for the classification recipe.

    The recordings are AVI videos, so `build()` extracts the audio as 16 kHz
    mono WAV and writes one manifest per split, drawn as an actor-held-out
    split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a UTeMo source tree is reachable."""
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
            logger.info("UTeMo already present; skipping download")
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
