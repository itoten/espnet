"""UA-SER dataset builder."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the UA-SER source tree, in order.

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


def _list_wavs(source_root: Path) -> list[Path]:
    """List the recordings in ``recordings_dir`` below ``source_root``."""
    return sorted((source_root / str(_CFG["recordings_dir"])).glob("*.wav"))


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the metadata file and all recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding ``metadata_file`` and at least
        ``expected_files`` recordings in ``recordings_dir``.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and (candidate / str(_CFG["metadata_file"])).is_file()
            and len(_list_wavs(candidate)) >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete UA-SER data not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['metadata_file']}` and `{_CFG['recordings_dir']}/*.wav`, or let "
        "`prepare_source` download them from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$UA_SER_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _git_blob_oid(path: Path) -> str:
    """Return the git blob id of a file (what Hugging Face lists as ``oid``)."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _list_files() -> list[dict]:
    """List the files of the pinned Hugging Face revision.

    Raises:
        urllib.error.URLError: If the listing cannot be fetched.
        ValueError: If the number of recordings is not ``expected_files``
            (a shorter list would mean a truncated reply).
    """
    url = (
        f"https://huggingface.co/api/datasets/{_CFG['repo_id']}/tree/"
        f"{_CFG['revision']}?recursive=true"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        files = [e for e in json.load(response) if e["type"] == "file"]
    prefix = f"{_CFG['recordings_dir']}/"
    wavs = [e for e in files if e["path"].startswith(prefix)]
    if len(wavs) != _CFG["expected_files"]:
        raise ValueError(
            f"Listed {len(wavs)} recordings, expected {_CFG['expected_files']}"
        )
    return [e for e in files if e in wavs or e["path"] == _CFG["metadata_file"]]


def _fetch_file(entry: dict, destination: Path) -> None:
    """Download one file and check it against the listing.

    Recordings are checked by size and SHA-256 (``lfs.oid``), the metadata file
    by its git blob id. The target path is built from the listing's relative
    path, which must stay inside ``destination``.

    Raises:
        ValueError: If the path escapes ``destination`` or the file does not
            match the listing.
    """
    target = (destination / entry["path"]).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"{entry['path']} escapes {destination}")
    if target.is_file():
        return
    url = (
        f"https://huggingface.co/datasets/{_CFG['repo_id']}/resolve/"
        f"{_CFG['revision']}/{urllib.parse.quote(entry['path'])}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as dst:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            dst.write(chunk)
    if "lfs" in entry:
        ok = (
            part.stat().st_size == entry["lfs"]["size"]
            and digest.hexdigest() == entry["lfs"]["oid"]
        )
    else:
        ok = _git_blob_oid(part) == entry["oid"] == _CFG["metadata_oid"]
    if not ok:
        raise ValueError(f"{entry['path']}: does not match the listing")
    part.replace(target)


def _download_source(destination: Path) -> None:
    """Fetch the metadata file and every recording of the pinned revision.

    Args:
        destination: Directory that will hold both.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a file does not match the listing.
    """
    destination.mkdir(parents=True, exist_ok=True)
    files = _list_files()
    logger.info("Downloading %d files to %s", len(files), destination)
    with ThreadPoolExecutor(max_workers=int(_CFG["num_workers"])) as pool:
        list(pool.map(lambda entry: _fetch_file(entry, destination), files))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the metadata file and pair each row with its recording.

    Args:
        source_root: Corpus root holding the metadata file and recordings.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker``,
        ``label`` and ``split`` (the published ``train`` or ``test``).

    Raises:
        ValueError: If a row has an unknown split, a repeated file name, or
            no recording, or a name in ``duplicate_files`` is not found.
    """
    excluded = (
        {str(n) for n in _CFG["duplicate_files"]}
        if _CFG["exclude_duplicates"]
        else set()
    )
    entries = []
    seen = set()
    with (source_root / str(_CFG["metadata_file"])).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["filename"]
            wav = source_root / str(_CFG["recordings_dir"]) / name
            if row["split"] not in ("train", "test"):
                raise ValueError(f"Unknown split {row['split']!r} for {name}")
            if name in seen:
                raise ValueError(f"Duplicate row for {name}")
            if not wav.is_file():
                raise ValueError(f"{name} is listed but missing: {wav}")
            seen.add(name)
            if name in excluded:
                excluded.discard(name)
                continue
            entries.append(
                {
                    "utt_id": f"ua_ser-{Path(name).stem}",
                    "source": str(wav),
                    "speaker": row["speaker_id"],
                    "label": row["emotion"],
                    "split": row["split"],
                }
            )
    if excluded:
        raise ValueError(f"`duplicate_files` names not found: {sorted(excluded)}")
    if not entries:
        raise ValueError("The metadata file has no rows")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Keep the published train/test split and cut valid out of train by speaker.

    A ``valid_speaker_ratio`` share of the train speakers (shuffled with
    ``split_seed``) is moved to valid, so no speaker is in two splits.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If a speaker is in both the published train and test.
    """
    train_speakers = {e["speaker"] for e in entries if e["split"] == "train"}
    test_speakers = {e["speaker"] for e in entries if e["split"] == "test"}
    if train_speakers & test_speakers:
        raise ValueError(
            "Speakers in both train and test: "
            f"{sorted(train_speakers & test_speakers)}"
        )
    speakers = sorted(train_speakers)
    Random(int(_CFG["split_seed"])).shuffle(speakers)
    n_valid = round(len(speakers) * float(_CFG["valid_speaker_ratio"]))
    valid_speakers = set(speakers[:n_valid])
    return {
        e["utt_id"]: (
            "test"
            if e["split"] == "test"
            else "valid" if e["speaker"] in valid_speakers else "train"
        )
        for e in entries
    }


class UaSerBuilder(DatasetBuilder):
    """Prepare UA-SER for the classification recipe.

    `build()` converts every recording to 16 kHz mono and writes one manifest
    per split, following the published speaker-disjoint train/test split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a UA-SER source tree is reachable."""
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
        """Download the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("UA-SER already present; skipping download")
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
