"""USC-EMO-MRI dataset builder (Zenodo archive, speaker-held-out split)."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

UTT_ID_RE = re.compile(r"_\d{3}$")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the USC-EMO-MRI tree, in order.

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
        Path: Corpus root with the speaker directories (``f1`` ... ``m5``) below it.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_avi`` AVI files.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and sum(1 for _ in candidate.rglob("*.avi")) >= _CFG["expected_avi"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete USC-EMO-MRI data not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to the extracted directory, or let "
        "`prepare_source` download it from Zenodo."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$USC_EMO_MRI_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_members(archive: Path, destination: Path) -> None:
    """Extract the AVI and label files of the archive into ``destination``.

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
            if member.startswith("__MACOSX/") or member.endswith("/"):
                continue
            if not (member.lower().endswith(".avi") or "/labels/" in member):
                continue
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member}")
            zf.extract(member, destination)


def _download_source(destination: Path) -> None:
    """Fetch the archive and unpack the AVI and label files.

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
    _extract_members(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def _read_table(path: Path) -> list[list[str]]:
    """Return the tab-separated rows of ``path`` without the header line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split("\t") for line in lines[1:] if line.strip()]


def read_entries(source_root: Path) -> list[dict]:
    """Collect the labelled utterances of every speaker.

    Utterances with no ``meta`` row (no label) or with an emotion ID outside
    ``emotion_map`` are dropped and counted in the log.

    Args:
        source_root: Directory holding the speaker directories somewhere below it.

    Returns:
        list[dict]: Keys ``utt_id``, ``speaker``, ``avi``, ``start``, ``end``
        and ``label``.

    Raises:
        ValueError: If an utterance id repeats or its AVI file is missing.
        FileNotFoundError: If no utterance is found.
    """
    entries = []
    dropped: Counter = Counter()
    for label_dir in sorted(source_root.rglob("labels")):
        speaker = label_dir.parent.name
        emotion_of = {row[0]: row[1] for row in _read_table(label_dir / "meta")}
        for row in _read_table(label_dir / "utt_time_stamp"):
            # One speaker's file has the id and AVI-name columns swapped.
            utt, base = (
                (row[0], row[1]) if UTT_ID_RE.search(row[0]) else (row[1], row[0])
            )
            if utt not in emotion_of:
                dropped["no label"] += 1
                continue
            if emotion_of[utt] not in _EMOTION_MAP:
                dropped[f"emotion ID {emotion_of[utt]}"] += 1
                continue
            avi = label_dir.parent / "avi" / f"{base}_withaudio.avi"
            if not avi.is_file():
                raise ValueError(f"AVI file missing for {utt}: {avi}")
            entries.append(
                {
                    "utt_id": f"usc-emo-mri-{speaker}-{utt}",
                    "speaker": speaker,
                    "avi": str(avi),
                    "start": row[2],
                    "end": row[3],
                    "label": _EMOTION_MAP[emotion_of[utt]],
                }
            )
    if dropped:
        logger.warning("Dropped utterances: %s", dict(dropped))
    if not entries:
        raise FileNotFoundError(f"No USC-EMO-MRI utterance under {source_root}")
    names = Counter(e["utt_id"] for e in entries)
    duplicated = [name for name, n in names.items() if n > 1]
    if duplicated:
        raise ValueError(f"Utterance ids are not unique, e.g. {duplicated[:3]}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict]) -> dict[str, str]:
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


class UscEmoMriBuilder(DatasetBuilder):
    """Prepare USC-EMO-MRI for the classification recipe.

    `build()` cuts every labelled utterance out of its AVI as 16 kHz mono
    audio and writes one manifest per split, drawn as a speaker-held-out
    split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a USC-EMO-MRI source tree is reachable."""
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
            logger.info("USC-EMO-MRI already present; skipping download")
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
        """Cut and resample the utterances and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a conversion fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required to extract the audio")

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
                        entry["avi"],
                        "-ss",
                        entry["start"],
                        "-to",
                        entry["end"],
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
