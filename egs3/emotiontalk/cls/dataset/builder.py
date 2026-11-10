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


def _extraction_unfinished(candidate: Path) -> bool:
    """Tell whether an extraction into ``candidate`` was interrupted.

    With ``remove_archive`` the archive is deleted only after it has been
    extracted completely, so an archive still next to ``Audio`` means the
    extraction stopped part-way (for example, the job was killed). Without
    ``remove_archive`` this cannot be told and ``False`` is returned.
    """
    return bool(_CFG["remove_archive"]) and (candidate / _CFG["archive_name"]).is_file()


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


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the extracted labels and audio.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding ``Audio/json`` and ``Audio/wav``.

    Raises:
        FileNotFoundError: If no candidate holds both directories, or the one
            that does still holds the archive (see :func:`_extraction_unfinished`).
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (candidate / _CFG["json_dir"]).is_dir() and (
            candidate / _CFG["wav_dir"]
        ).is_dir():
            if not _extraction_unfinished(candidate):
                return candidate
            logger.warning(
                "%s still holds %s: the extraction did not finish",
                candidate,
                _CFG["archive_name"],
            )
        checked.append(str(candidate))
    raise FileNotFoundError(
        "EmotionTalk audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the extracted "
        f"`Audio` folder, or let `prepare_source` download `{_CFG['archive_name']}`."
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


def _extract_archive(archive: Path, source_root: Path) -> None:
    """Extract the archive into ``source_root`` and optionally delete it."""
    with tarfile.open(archive) as tar:
        tar.extractall(source_root, filter="data")
    if _CFG["remove_archive"]:
        archive.unlink()


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


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the label files and match each to its audio file.

    Args:
        source_root: Directory returned by :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker``,
        ``split`` and ``label``, sorted by id.

    Raises:
        ValueError: If a label is unknown or there are too few utterances.
        FileNotFoundError: If an audio file is missing.
    """
    valid_groups = {str(g) for g in _CFG["valid_groups"]}
    test_groups = {str(g) for g in _CFG["test_groups"]}
    json_root = source_root / _CFG["json_dir"]
    wav_root = source_root / _CFG["wav_dir"]

    entries = []
    for path in sorted(json_root.glob("*/*/*/*.json")):
        with path.open(encoding="utf-8") as fh:
            row = json.load(fh)
        emotion = row["emotion_result"]
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {emotion!r}. Known: {sorted(_EMOTION_MAP)}"
            )
        wav = wav_root / row["file_path"]
        if not wav.is_file():
            raise FileNotFoundError(f"Audio listed in the labels is missing: {wav}")
        group = path.relative_to(json_root).parts[0]
        split = "train"
        if group in valid_groups:
            split = "valid"
        elif group in test_groups:
            split = "test"
        entries.append(
            {
                "utt_id": f"emotiontalk-{path.stem}",
                "source": str(wav),
                "speaker": str(row["speaker_id"]),
                "split": split,
                "label": _EMOTION_MAP[emotion],
            }
        )
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

    `prepare_source` downloads and extracts ``Audio.tar`` (the Hugging Face
    dataset is gated; see ``readme.md``). `build()` converts every utterance
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
        """Download and extract the audio unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            PermissionError: If the account has not agreed to the dataset's terms.
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("EmotionTalk already present; skipping download")
            return
        destination = next(iter_source_candidates(recipe_root, source_dir))
        archive = _download_archive(destination)
        _extract_archive(archive, destination)
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
        """Resample the utterances and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location of the EmotionTalk files.

        Raises:
            FileNotFoundError: If the files or ``ffmpeg`` cannot be found.
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

        entries = read_entries(source_root)
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
