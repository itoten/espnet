"""EmotionTalk dataset builder."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
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
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the EmotionTalk files, in order.

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


def _complete_archive(candidate: Path) -> Path | None:
    """Return ``candidate/Audio.tar`` if it has its full size, else None."""
    archive = candidate / _CFG["archive_name"]
    if archive.is_file() and archive.stat().st_size == int(_CFG["archive_size"]):
        return archive
    return None


def _has_extracted_tree(candidate: Path) -> bool:
    return (candidate / _CFG["json_dir"]).is_dir() and (
        candidate / _CFG["wav_dir"]
    ).is_dir()


def resolve_source(recipe_root: Path, source_dir: str | Path | None = None) -> Path:
    """Return the EmotionTalk audio to read: the archive or an extracted tree.

    In each candidate directory a complete ``Audio.tar`` is preferred, read in
    place without extracting it; otherwise an already extracted ``Audio``
    folder (``Audio/json`` and ``Audio/wav``) is used.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: The ``Audio.tar`` file, or the directory holding ``Audio``.

    Raises:
        FileNotFoundError: If no candidate holds either.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        archive = _complete_archive(candidate)
        if archive is not None:
            return archive
        if _has_extracted_tree(candidate):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "EmotionTalk audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['archive_name']}` (or the extracted `Audio` folder), or let "
        "`prepare_source` download it."
    )


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        return token_file.read_text().strip() or None
    return None


def _download_archive(destination: Path) -> Path:
    """Fetch ``Audio.tar`` from the gated Hugging Face dataset.

    A partial download is resumed.

    Raises:
        PermissionError: If the account has not agreed to the dataset's terms.
        ValueError: If the file does not reach its listed size.
    """
    name = str(_CFG["archive_name"])
    size = int(_CFG["archive_size"])
    target = destination / name
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{_CFG['hf_repo']}/resolve/main/{name}"
    for _ in range(5):
        have = target.stat().st_size if target.is_file() else 0
        if have == size:
            return target
        headers = {"Range": f"bytes={have}-"} if have else {}
        token = _hf_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as response:
                with target.open("ab" if have else "wb") as fh:
                    shutil.copyfileobj(response, fh, 1 << 20)
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                raise PermissionError(
                    f"Access to {_CFG['hf_repo']} was refused. Sign in to "
                    "Hugging Face, open https://huggingface.co/datasets/"
                    f"{_CFG['hf_repo']} and agree to share your username and "
                    "email with the authors, then make the account's token "
                    "available (`huggingface-cli login` or $HF_TOKEN)."
                ) from err
            raise
        except OSError as err:
            logger.warning("Retrying %s: %s", name, err)
    if target.stat().st_size != size:
        raise ValueError(f"Incomplete download: {name}")
    return target


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMOTIONTALK_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _scan_archive(archive: Path) -> tuple[list[tuple[str, dict]], dict[str, tuple[int, int]]]:
    """Read the label files and locate the audio files inside ``Audio.tar``.

    The archive is not compressed, so each member's bytes sit at a fixed
    offset and can be read later without extracting anything.

    Returns:
        The labels as ``(member name, parsed JSON)`` and, for each audio
        member name, its ``(offset, size)`` in the archive.
    """
    json_prefix = f"{_CFG['json_dir']}/"
    wav_prefix = f"{_CFG['wav_dir']}/"
    labels: list[tuple[str, dict]] = []
    audio: dict[str, tuple[int, int]] = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name.startswith(json_prefix) and member.name.endswith(".json"):
                with tar.extractfile(member) as fh:
                    labels.append((member.name[len(json_prefix):], json.load(fh)))
            elif member.name.startswith(wav_prefix):
                audio[member.name[len(wav_prefix):]] = (member.offset_data, member.size)
    return labels, audio


def _scan_tree(source_root: Path) -> list[tuple[str, dict]]:
    """Read the label files of an extracted ``Audio`` folder."""
    json_root = source_root / _CFG["json_dir"]
    labels = []
    for path in json_root.glob("*/*/*/*.json"):
        with path.open(encoding="utf-8") as fh:
            labels.append((path.relative_to(json_root).as_posix(), json.load(fh)))
    return labels


def read_entries(source: Path) -> list[dict]:
    """Read the label files and match each to its audio.

    Args:
        source: Path returned by :func:`resolve_source`: ``Audio.tar`` or a
            directory holding the extracted ``Audio`` folder.

    Returns:
        list[dict]: Keys ``utt_id``, ``speaker``, ``split``, ``label`` and the
        audio: ``source`` (a file path) for an extracted folder, or
        ``archive``, ``offset`` and ``size`` for the archive. Sorted by id.

    Raises:
        ValueError: If a label is unknown or there are too few utterances.
        FileNotFoundError: If an audio file is missing.
    """
    valid_groups = {str(g) for g in _CFG["valid_groups"]}
    test_groups = {str(g) for g in _CFG["test_groups"]}
    in_archive = source.is_file()
    if in_archive:
        labels, audio = _scan_archive(source)
    else:
        labels = _scan_tree(source)
        wav_root = source / _CFG["wav_dir"]

    entries = []
    for json_name, row in sorted(labels):
        emotion = row["emotion_result"]
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {emotion!r}. Known: {sorted(_EMOTION_MAP)}"
            )
        group = json_name.split("/")[0]
        split = "train"
        if group in valid_groups:
            split = "valid"
        elif group in test_groups:
            split = "test"
        entry = {
            "utt_id": f"emotiontalk-{Path(json_name).stem}",
            "speaker": str(row["speaker_id"]),
            "split": split,
            "label": _EMOTION_MAP[emotion],
        }
        if in_archive:
            location = audio.get(row["file_path"])
            if location is None:
                raise FileNotFoundError(
                    f"Audio listed in the labels is missing from {source}: "
                    f"{_CFG['wav_dir']}/{row['file_path']}"
                )
            entry.update(archive=str(source), offset=location[0], size=location[1])
        else:
            wav = wav_root / row["file_path"]
            if not wav.is_file():
                raise FileNotFoundError(f"Audio listed in the labels is missing: {wav}")
            entry["source"] = str(wav)
        entries.append(entry)
    if len(entries) < int(_CFG["expected_utterances"]):
        raise ValueError(
            f"Found {len(entries)} utterances, expected {_CFG['expected_utterances']}"
        )
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


class EmotiontalkBuilder(DatasetBuilder):
    """Prepare EmotionTalk for the classification recipe.

    `prepare_source` downloads ``Audio.tar`` (the Hugging Face dataset is
    gated; see ``readme.md``) and keeps it as it is. `build()` reads the
    labels and the audio straight from the archive, converts every utterance
    to 16 kHz mono and writes one manifest per split, following the published
    speaker-group partition.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether nothing is left to download: built data or raw audio."""
        if self.is_built(recipe_dir):
            return True
        try:
            resolve_source(Path(recipe_dir).resolve(), source_dir)
        except FileNotFoundError:
            return False
        return True

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Download ``Audio.tar`` unless the audio is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            PermissionError: If the account has not agreed to the dataset's terms.
            ValueError: If the download does not reach its listed size.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("EmotionTalk already present; skipping download")
            return
        destination = next(iter_source_candidates(recipe_root, source_dir))
        _download_archive(destination)

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
        """Resample the utterances and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location of the EmotionTalk files.

        Raises:
            FileNotFoundError: If the files or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a resample fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source = resolve_source(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required to resample the recordings to 16 kHz"
            )

        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)
        sampling_rate = str(_CFG["sampling_rate"])

        def convert(entry: dict) -> Path:
            wav = wav_dir / f"{entry['utt_id']}.wav"
            if wav.is_file():
                return wav
            if "archive" in entry:
                # Read the member's bytes from the archive and pipe them in.
                with open(entry["archive"], "rb") as fh:
                    fh.seek(entry["offset"])
                    data = fh.read(entry["size"])
                if len(data) != entry["size"]:
                    raise ValueError(f"Truncated archive: {entry['archive']}")
                source_arg, stdin = "pipe:0", data
            else:
                source_arg, stdin = entry["source"], None
            part = wav.with_suffix(wav.suffix + ".part")
            subprocess.run(
                [
                    ffmpeg,
                    "-i",
                    source_arg,
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
                input=stdin,
                check=True,
            )
            part.replace(wav)
            return wav

        entries = read_entries(source)
        with ThreadPoolExecutor(int(_CFG["convert_workers"])) as pool:
            wavs = list(pool.map(convert, entries))

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry, wav in zip(entries, wavs):
            rows[entry["split"]].append(
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
