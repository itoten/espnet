"""JL-Corpus dataset builder."""

from __future__ import annotations

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

logger = logging.getLogger(__name__)

# `female1_angry_10a_1.wav` -> speaker female1, emotion angry, sentence 10,
# session a, repetition 1. `Format_Intro.txt` states the rule as
# `(Gender)(speaker.ID)_(Emotion)_(Sentence.ID)(session.ID)`.
FILENAME_RE = re.compile(
    r"^(?P<speaker>(?:female|male)\d+)_(?P<emotion>[a-z]+)_"
    r"(?P<sentence>\d+)(?P<session>[a-z])_(?P<repetition>\d+)\.wav$"
)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _emotion_map() -> dict[str, str]:
    """Return the emotions to keep, mapped to their manifest labels.

    Raises:
        ValueError: If ``emotion_set`` is not a known set.
    """
    primary = {str(k): str(v) for k, v in _CFG["primary_emotions"].items()}
    secondary = {str(k): str(v) for k, v in _CFG["secondary_emotions"].items()}
    choice = str(_CFG["emotion_set"])
    if choice == "all":
        return {**primary, **secondary}
    if choice == "primary":
        return primary
    if choice == "secondary":
        return secondary
    raise ValueError(
        f"Unknown `emotion_set` {choice!r}; use 'all', 'primary' or 'secondary'."
    )


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the JL-Corpus source tree, in order.

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


def resolve_audio_dir(root: Path) -> Path | None:
    """Return the directory holding the recordings under ``root``.

    The archive nests the audio a few levels down, but a copy is sometimes
    flattened, so both layouts are accepted.

    Args:
        root: Candidate corpus root.

    Returns:
        Path | None: Directory holding the WAV files, or None when absent.
    """
    for candidate in (root / str(_CFG["audio_subdir"]), root):
        if candidate.is_dir() and any(candidate.glob("*.wav")):
            return candidate
    return None


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate whose tree holds the recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding the WAV files.

    Raises:
        FileNotFoundError: If no candidate holds any.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        audio_dir = resolve_audio_dir(candidate)
        if audio_dir is not None:
            return audio_dir
        checked.append(str(candidate))
    raise FileNotFoundError(
        "JL-Corpus audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$JLCORPUS_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _download_source(destination: Path) -> None:
    """Fetch the corpus from Kaggle and unpack it.

    The GitHub repository carries only the supporting documents, so Kaggle is
    the only distribution point and an API token is required.

    Args:
        destination: Directory to unpack into.

    Raises:
        RuntimeError: If the Kaggle CLI is missing.
        subprocess.CalledProcessError: If the download fails.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The `kaggle` command is required to download JL-Corpus, which is "
            "published on Kaggle and nowhere else. Install it with "
            "`pip install kaggle`, then put an API token from "
            "https://www.kaggle.com/settings/api in ~/.kaggle/access_token or "
            "in $KAGGLE_API_TOKEN."
        )
    destination.mkdir(parents=True, exist_ok=True)
    dataset = str(_CFG["kaggle_dataset"])
    logger.info("Downloading %s from Kaggle (about 1.3 GB)", dataset)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(destination)],
        check=True,
    )

    archives = sorted(destination.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"Kaggle wrote no archive into {destination}")
    for archive in archives:
        _extract_zip(archive, destination)
        if _CFG.get("remove_archive", False):
            archive.unlink(missing_ok=True)


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


def parse_filename(name: str) -> dict[str, str] | None:
    """Split a JL-Corpus file name into its fields.

    Args:
        name: File name such as ``female1_angry_10a_1.wav``.

    Returns:
        dict[str, str] | None: Keys ``speaker``, ``emotion``, ``sentence``,
        ``session`` and ``repetition``, or None when the name does not match.

    Examples:
        >>> parse_filename("female1_angry_10a_1.wav")["emotion"]
        'angry'
    """
    match = FILENAME_RE.match(name)
    return match.groupdict() if match else None


def read_entries(audio_dir: Path) -> list[dict[str, str]]:
    """Collect the recordings whose emotion is in the configured set.

    Args:
        audio_dir: Directory holding the WAV files.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        FileNotFoundError: If no recording matches.
    """
    emotions = _emotion_map()
    entries = []
    skipped = 0
    for wav in sorted(audio_dir.glob("*.wav")):
        fields = parse_filename(wav.name)
        if fields is None:
            skipped += 1
            continue
        if fields["emotion"] not in emotions:
            continue
        entries.append(
            {
                "utt_id": f"jlcorpus-{wav.stem}",
                "source": str(wav),
                "speaker": fields["speaker"],
                "label": emotions[fields["emotion"]],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if not entries:
        raise FileNotFoundError(f"No JL-Corpus audio under {audio_dir}")
    logger.info(
        "Read %d utterance(s) for emotion_set=%s", len(entries), _CFG["emotion_set"]
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out the test speakers, then draw validation from the rest.

    Test stays speaker independent. Validation is drawn from the remaining
    speakers instead of holding out another one, because the corpus has only
    four: a second held-out speaker would leave two for training. **Validation
    therefore shares speakers with training** and reads higher than test.

    The draw is stratified by label and seeded, so it depends on the corpus and
    ``split_seed`` alone.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If a configured test speaker is not in the corpus, or if
            holding them out leaves nothing to train on.
    """
    test_speakers = {str(s) for s in _CFG["test_speakers"]}
    speakers = {e["speaker"] for e in entries}
    unknown = test_speakers - speakers
    if unknown:
        raise ValueError(
            f"`test_speakers` names {sorted(unknown)}, which the corpus does "
            f"not hold ({sorted(speakers)})."
        )
    if speakers == test_speakers:
        raise ValueError(
            "`test_speakers` lists every speaker in the corpus, leaving "
            "nothing to train on."
        )

    ratio = float(_CFG["valid_ratio"])
    rng = Random(int(_CFG["split_seed"]))

    by_label: dict[str, list[str]] = {}
    assignment = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        if entry["speaker"] in test_speakers:
            assignment[entry["utt_id"]] = "test"
        else:
            by_label.setdefault(entry["label"], []).append(entry["utt_id"])

    for _, utt_ids in sorted(by_label.items()):
        shuffled = list(utt_ids)
        rng.shuffle(shuffled)
        count = round(len(shuffled) * ratio)
        for utt_id in shuffled[:count]:
            assignment[utt_id] = "valid"
        for utt_id in shuffled[count:]:
            assignment[utt_id] = "train"
    return assignment


class JLCorpusBuilder(DatasetBuilder):
    """Prepare JL-Corpus for the classification recipe.

    The corpus ships 44.1 kHz mono WAV, so `build()` resamples every recording
    to 16 kHz and writes one manifest per split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a JL-Corpus source tree is reachable."""
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
            RuntimeError: If the Kaggle CLI is missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("JL-Corpus already present; skipping download")
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
        audio_dir = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required to resample the recordings to 16 kHz"
            )

        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)
        sampling_rate = str(_CFG["sampling_rate"])

        entries = read_entries(audio_dir)
        assignment = assign_splits(entries)

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
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
