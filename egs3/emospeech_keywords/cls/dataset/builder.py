"""EmoSpeech (keyword) dataset builder (manually obtained Kaggle archive)."""

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

logger = logging.getLogger(__name__)

# `bacho-safe-angry-1562907046662-<56 hex>.wav`: keyword, environment (may hold a
# space, e.g. `glass break`), emotion, time stamp (may hold a decimal point or a
# " copy" suffix) and the recording user's hash.
FILENAME_RE = re.compile(
    r"^(?P<keyword>[^-]+)-(?P<environment>.+)-(?P<emotion>[A-Za-z]+)-"
    r"(?P<stamp>\d+(?:\.\d+)?(?: copy)?)-(?P<user>[0-9a-f]{56})\.wav$"
)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k).lower(): str(v) for k, v in _CFG["emotion_map"].items()}
_DROP_CODES = {str(c).lower() for c in _CFG["drop_codes"]}
_DROP_KEYWORDS = {str(k).lower() for k in _CFG["drop_keywords"]}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the extracted archive, in order.

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

    Recordings are searched recursively, since they sit in one directory per
    recording location.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding all recordings below it.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and sum(1 for _ in candidate.rglob("*.wav")) >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete EmoSpeech (keyword) audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + "\nLet `prepare_source` download it with the Kaggle CLI (an API token "
        "in ~/.kaggle/access_token or $KAGGLE_API_TOKEN), or download the "
        f"archive from {_CFG['kaggle_url']} (a free Kaggle "
        "account is needed), extract it, and set "
        f"${_CFG['source_env_var']} to the extracted directory."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMOSPEECH_KEYWORDS_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the file name.

    Recordings with the same audio content are kept once (the first path in
    sorted order), and recordings labelled with a ``drop_codes`` value or
    holding a ``drop_keywords`` keyword are skipped. Both counts are logged.

    Args:
        source_root: Corpus root with the recordings somewhere below it.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` and ``label``.

    Raises:
        ValueError: If a file name does not parse, an emotion is unknown, or
            two files map to the same utterance id.
        FileNotFoundError: If no recording is left.
    """
    entries = []
    seen_ids: dict[str, Path] = {}
    seen_audio: dict[str, Path] = {}
    n_duplicate = 0
    n_dropped = 0
    for wav in sorted(source_root.rglob("*.wav")):
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            raise ValueError(f"Unexpected file name: {wav}")
        emotion = fields["emotion"].lower()
        if emotion in _DROP_CODES or fields["keyword"].lower() in _DROP_KEYWORDS:
            n_dropped += 1
            continue
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {fields['emotion']!r} in {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        digest = hashlib.md5(wav.read_bytes()).hexdigest()
        if _CFG["exclude_duplicates"] and digest in seen_audio:
            n_duplicate += 1
            continue
        seen_audio[digest] = wav
        stem = f"{wav.parent.name}-{wav.stem}".replace(" ", "_")
        if stem in seen_ids:
            raise ValueError(f"Duplicate utterance id: {wav} and {seen_ids[stem]}")
        seen_ids[stem] = wav
        entries.append(
            {
                "utt_id": f"emospeech_keywords-{stem}",
                "source": str(wav),
                "label": _EMOTION_MAP[emotion],
            }
        )
    logger.info(
        "Read %d utterance(s); skipped %d duplicate recording(s) and %d with a "
        "dropped label",
        len(entries),
        n_duplicate,
        n_dropped,
    )
    if not entries:
        raise FileNotFoundError(f"No EmoSpeech (keyword) audio under {source_root}")
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. Speakers are not held out, so the
    splits share speakers.

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

    rng = Random(int(_CFG["split_seed"]))
    by_label: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        by_label.setdefault(entry["label"], []).append(entry["utt_id"])

    assignment: dict[str, str] = {}
    for _, utt_ids in sorted(by_label.items()):
        shuffled = list(utt_ids)
        rng.shuffle(shuffled)
        n_valid = round(len(shuffled) * valid_ratio)
        n_test = round(len(shuffled) * test_ratio)
        for utt_id in shuffled[:n_test]:
            assignment[utt_id] = "test"
        for utt_id in shuffled[n_test : n_test + n_valid]:
            assignment[utt_id] = "valid"
        for utt_id in shuffled[n_test + n_valid :]:
            assignment[utt_id] = "train"
    return assignment


def _download_source(destination: Path) -> None:
    """Fetch the corpus from Kaggle and unpack it into ``destination``.

    Kaggle is the only distribution point, so the Kaggle CLI and an API token
    are required, as in ``egs3/jlcorpus/cls``.

    Args:
        destination: Directory to unpack into.

    Raises:
        RuntimeError: If the Kaggle CLI is missing.
        subprocess.CalledProcessError: If the download fails.
        FileNotFoundError: If Kaggle wrote no archive.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The `kaggle` command is required to download EmoSpeech (keyword), which is "
            "published on Kaggle. Install it with `pip install kaggle`, then put "
            "an API token from https://www.kaggle.com/settings/api in "
            "~/.kaggle/access_token or in $KAGGLE_API_TOKEN."
        )
    destination.mkdir(parents=True, exist_ok=True)
    dataset = str(_CFG["kaggle_dataset"])
    logger.info("Downloading %s from Kaggle", dataset)
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

    Only members starting with ``extract_prefix`` and ending with
    ``extract_suffix`` (both optional in ``dataset/config.yaml``) are written.
    Members are checked against the destination first: ``ZipFile.extractall``
    does not reject absolute paths or ``..`` components on its own.

    Args:
        archive: Archive to read.
        destination: Directory to extract into.

    Raises:
        ValueError: If a member would be written outside ``destination``.
        zipfile.BadZipFile: If the archive is invalid.
    """
    prefix = str(_CFG.get("extract_prefix") or "")
    suffix = str(_CFG.get("extract_suffix") or "").lower()
    logger.info("Extracting %s", archive.name)
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = [
            name
            for name in zf.namelist()
            if name.startswith(prefix) and name.lower().endswith(suffix)
        ]
        for name in members:
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError(f"{archive.name} escapes its destination: {name}")
        zf.extractall(path=destination, members=members)


class EmospeechKeywordsBuilder(DatasetBuilder):
    """Prepare EmoSpeech (keyword) for the classification recipe.

    `build()` copies every recording (already 16 kHz mono) into the shared
    layout and writes one manifest per split, drawn as a label-stratified random
    split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a EmoSpeech (keyword) source tree is reachable."""
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
            logger.info("EmoSpeech (keyword) already present; skipping download")
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
