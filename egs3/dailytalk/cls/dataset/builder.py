"""DailyTalk dataset builder (Google Drive zip, dialogue-level split)."""

from __future__ import annotations

import hashlib
import json
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
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `data/<dialog>/<utterance>_<speaker>_d<dialog>.wav`, e.g. `data/0/10_1_d0.wav`.
FILENAME_RE = re.compile(r"^(?P<utt>\d+)_(?P<speaker>\d)_d(?P<dialog>\d+)\.wav$")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the DailyTalk source tree, in order.

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


def _find_root(candidate: Path) -> Path | None:
    """Return the directory holding ``metadata.json`` at or just below candidate."""
    for root in (candidate, candidate / "dailytalk"):
        if (root / "metadata.json").is_file():
            return root
    return None


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the complete set of recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with ``metadata.json`` and ``data/`` below it.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        root = _find_root(candidate) if candidate.is_dir() else None
        if (
            root is not None
            and sum(1 for wav in (root / "data").rglob("*.wav"))
            >= _CFG["expected_files"]
        ):
            return root
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete DailyTalk audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "extracted `metadata.json` and `data/`, or let `prepare_source` download "
        "them."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$DAILYTALK_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_source(archive: Path, destination: Path) -> None:
    """Extract the WAV files and ``metadata.json`` of the archive.

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
            if not member.lower().endswith((".wav", "metadata.json")):
                continue
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member}")
            zf.extract(member, destination)


def _download_source(destination: Path) -> None:
    """Fetch the archive and unpack the recordings and labels.

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
    _extract_source(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Pair every recording with its label from ``metadata.json``.

    Args:
        source_root: Corpus root with ``metadata.json`` and ``data/``.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker``,
        ``dialog`` and ``label``.

    Raises:
        ValueError: If an emotion is unknown, a recording disagrees with the
            metadata on dialogue or speaker, or an `duplicate_files` name is absent.
        FileNotFoundError: If a recording has no metadata entry or none is found.
    """
    metadata = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    entries = []
    skipped = 0
    excluded = (
        set(_CFG["duplicate_files"]) if _CFG["exclude_duplicates"] else set()
    )
    for wav in sorted((source_root / "data").rglob("*.wav")):
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            skipped += 1
            continue
        dialog, utt = fields["dialog"], fields["utt"]
        try:
            meta = metadata[dialog][utt]
        except KeyError:
            raise FileNotFoundError(f"No metadata entry for {wav}") from None
        if str(meta["speaker"]) != fields["speaker"]:
            raise ValueError(f"Speaker in the file name and metadata differ: {wav}")
        if meta["emotion"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {meta['emotion']!r} in {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        utt_id = f"dailytalk-d{int(dialog):04d}-u{int(utt):03d}"
        if utt_id in excluded:
            excluded.discard(utt_id)
            continue
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "speaker": fields["speaker"],
                "dialog": dialog,
                "label": _EMOTION_MAP[meta["emotion"]],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if excluded:
        raise ValueError(f"`duplicate_files` names not found: {sorted(excluded)}")
    if not entries:
        raise FileNotFoundError(f"No DailyTalk audio under {source_root}")
    names = Counter(e["utt_id"] for e in entries)
    duplicated = [name for name, n in names.items() if n > 1]
    if duplicated:
        raise ValueError(f"Utterance ids are not unique, e.g. {duplicated[:3]}")
    logger.info(
        "Read %d utterance(s) from %d dialogue(s)",
        len(entries),
        len({e["dialog"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Assign whole dialogues to splits at random with ``split_seed``.

    Both speakers take part in every dialogue, so speakers cannot be held out;
    keeping a dialogue in one split avoids sharing its context across splits.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.
    """
    dialogs = sorted({e["dialog"] for e in entries}, key=int)
    Random(int(_CFG["split_seed"])).shuffle(dialogs)
    n_test = round(len(dialogs) * float(_CFG["test_ratio"]))
    n_valid = round(len(dialogs) * float(_CFG["valid_ratio"]))
    split_of = {d: "test" for d in dialogs[:n_test]}
    split_of.update({d: "valid" for d in dialogs[n_test : n_test + n_valid]})
    return {
        e["utt_id"]: split_of.get(e["dialog"], "train")
        for e in sorted(entries, key=lambda e: e["utt_id"])
    }


class DailytalkBuilder(DatasetBuilder):
    """Prepare DailyTalk for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one manifest
    per split, drawn as a dialogue-level random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a DailyTalk source tree is reachable."""
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
            logger.info("DailyTalk already present; skipping download")
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

        def convert(entry: dict[str, str]) -> Path:
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
            return wav

        with ThreadPoolExecutor(max_workers=8) as pool:
            wavs = list(pool.map(convert, entries))

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry, wav in zip(entries, wavs):
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
