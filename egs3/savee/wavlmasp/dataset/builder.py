"""SAVEE dataset builder."""

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

# `DC_a01.wav` -> actor DC, emotion code `a`, index 01.
FILENAME_RE = re.compile(
    r"^(?P<actor>[A-Za-z]{2})_(?P<emotion>n|a|d|f|h|sa|su)(?P<index>\d+)\.wav$",
    re.IGNORECASE,
)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k).lower(): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the SAVEE source tree, in order.

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
    """Return the first candidate holding SAVEE WAV files directly.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory directly holding the WAV files.

    Raises:
        FileNotFoundError: If no candidate holds any.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir() and any(candidate.glob("*.wav")):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "SAVEE audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root holding "
        "`<actor>_<emotion><index>.wav` files, or place a hand-downloaded "
        "zip there for `prepare_source` to extract."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$SAVEE_OUTPUT`` when set, else ``<recipe_dir>/data``.
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
    """Return a hand-placed zip archive in any of the candidate directories.

    The download page's own archive naming is not documented anywhere
    reachable from a script (see ``dataset/config.yaml``), so this looks for
    any single zip rather than a fixed name.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path | None: The archive, or None when none is there.

    Raises:
        ValueError: If a candidate directory holds more than one zip.
    """
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if not candidate.is_dir():
            continue
        archives = sorted(candidate.glob(str(_CFG["archive_glob"])))
        if len(archives) > 1:
            raise ValueError(
                f"Found {len(archives)} zip files under {candidate}; "
                "remove all but the SAVEE download so `prepare_source` knows "
                "which one to extract."
            )
        if archives:
            return archives[0]
    return None


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking speaker and label from the file name.

    Args:
        source_root: Directory holding the WAV files.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        FileNotFoundError: If no recording is found.
    """
    entries = []
    skipped = 0
    for wav in sorted(source_root.glob("*.wav")):
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            skipped += 1
            continue
        emotion = fields["emotion"].lower()
        if emotion not in _EMOTION_MAP:
            skipped += 1
            continue
        entries.append(
            {
                "utt_id": f"savee-{wav.stem}",
                "source": str(wav),
                "speaker": fields["actor"].upper(),
                "label": _EMOTION_MAP[emotion],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if not entries:
        raise FileNotFoundError(f"No SAVEE audio under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
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


class SAVEEBuilder(DatasetBuilder):
    """Prepare SAVEE for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one
    manifest per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a SAVEE source tree is reachable."""
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
        """Unpack a hand-placed archive unless the tree is already there.

        SAVEE needs free registration to download
        (https://cvssp.org/savee/Download.html), so this cannot fetch it
        itself; it only extracts a zip the user already placed under
        ``$SAVEE`` or ``download/``.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If neither the tree nor an archive is present.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("SAVEE already present; skipping extraction")
            return

        archive = find_archive(recipe_root, source_dir)
        if archive is None:
            raise FileNotFoundError(
                "No SAVEE zip found in any of "
                + ", ".join(
                    str(c) for c in iter_source_candidates(recipe_root, source_dir)
                )
                + ".\nRegister and download it from "
                "https://cvssp.org/savee/Download.html, then place the zip "
                f"under ${_CFG['source_env_var']} (or `download/` under this "
                "recipe); this stage unpacks it from there."
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
