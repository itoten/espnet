"""Lithuanian Emotional Speech Database builder (GitHub archive)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tarfile
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `EMOdb/<speaker> Zmogus/<speaker> <Emotion>/<speaker> <Emotion>_<phrase>.wav`;
# a few files have a space before the underscore (`1 Baime _07.wav`).
FILENAME_RE = re.compile(
    r"^(?P<speaker>\d+) (?P<emotion>[A-Za-z]+) ?_(?P<phrase>\d+)\.wav$"
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
    """Yield the directories that may hold the corpus source tree, in order.

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
    """Return the first candidate holding the full set of recordings.

    Recordings are searched recursively, since they sit in
    `EMOdb/<speaker> Zmogus/<speaker> <Emotion>/` directories.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding all
        recordings below it.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and sum(
                1 for wav in candidate.rglob("*.wav") if FILENAME_RE.match(wav.name)
            )
            >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete Lithuanian emotion audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "extracted `EMOdb/` directory, or let `prepare_source` download it from "
        "GitHub."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$LITHUANIAN_EMOTION_DB_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_wavs(archive: Path, destination: Path) -> None:
    """Extract the WAV files of the archive into ``destination``.

    Args:
        archive: ``.tar.gz`` archive to read.
        destination: Directory to extract into.

    Raises:
        tarfile.TarError: If the archive is invalid.
        ValueError: If a member would be written outside ``destination``.
    """
    logger.info("Extracting %s", archive.name)
    root = destination.resolve()
    prefix = str(_CFG["archive_member_prefix"])
    with tarfile.open(archive) as tf:
        for member in tf:
            if not (member.isfile() and member.name.lower().endswith(".wav")):
                continue
            # Drop the leading `<repo>-<commit>/` directory.
            relative = member.name.split("/", 1)[1]
            if not relative.startswith(prefix):
                continue
            target = root / relative
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _download_source(destination: Path) -> None:
    """Fetch the archive and unpack its WAV files.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        tarfile.TarError: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = str(_CFG["archive_name"])

    archive_path = destination / archive_name
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    download_url(str(_CFG["archive_url"]), part, logger=logger)
    part.rename(archive_path)
    _extract_wavs(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the emotion folder.

    Args:
        source_root: Corpus root with the recordings somewhere below it.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a file name, its emotion folder or its speaker folder
            disagree, an emotion is unknown, or two recordings get the same id.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: dict[str, Path] = {}
    skipped = 0
    for wav in sorted(source_root.rglob("*.wav")):
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            skipped += 1
            continue
        speaker = fields["speaker"]
        emotion = fields["emotion"]
        folder_speaker, _, folder_emotion = wav.parent.name.partition(" ")
        if (folder_speaker, folder_emotion) != (speaker, emotion):
            raise ValueError(f"File name and folder disagree: {wav}")
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {emotion!r} in {wav}. Known: {sorted(_EMOTION_MAP)}"
            )
        utt_id = f"lithuanian-s{int(speaker):02d}-{emotion}-{fields['phrase']}"
        if utt_id in seen:
            raise ValueError(f"Duplicate recording id: {wav} and {seen[utt_id]}")
        seen[utt_id] = wav
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "speaker": speaker,
                "label": _EMOTION_MAP[emotion],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if not entries:
        raise FileNotFoundError(f"No Lithuanian emotion audio under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole speakers: ``n_test_speakers``, ``n_valid_speakers``, rest train.

    Speakers are shuffled with ``split_seed``, so every split has speakers the
    others never see.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If fewer speakers exist than the two held-out splits need
            plus one for training.
    """
    n_test = int(_CFG["n_test_speakers"])
    n_valid = int(_CFG["n_valid_speakers"])
    speakers = sorted({e["speaker"] for e in entries})
    if len(speakers) < n_test + n_valid + 1:
        raise ValueError(f"Only {len(speakers)} speakers for {n_test}+{n_valid}+train")
    Random(int(_CFG["split_seed"])).shuffle(speakers)
    split_of = {s: "test" for s in speakers[:n_test]}
    split_of.update({s: "valid" for s in speakers[n_test : n_test + n_valid]})
    return {
        e["utt_id"]: split_of.get(e["speaker"], "train")
        for e in sorted(entries, key=lambda e: e["utt_id"])
    }


class LithuanianEmotionDbBuilder(DatasetBuilder):
    """Prepare the Lithuanian emotion database for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes
    one manifest per split, drawn as a speaker-held-out split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the corpus source tree is reachable."""
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
            logger.info("Lithuanian emotion audio already present; skipping download")
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
