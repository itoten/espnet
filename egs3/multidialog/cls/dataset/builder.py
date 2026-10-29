"""MultiDialog dataset builder."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tarfile
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
_MERGE_LABELS = {str(k): str(v) for k, v in _CFG["merge_labels"].items()}
_SPLIT_MAP = {str(k): str(v) for k, v in _CFG["split_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the MultiDialog files, in order.

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
    """Return the first candidate holding the metadata and the extracted audio.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding ``metadata.jsonl`` and ``audio/<split>/``.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (candidate / _CFG["metadata_file"]).is_file() and (
            candidate / _CFG["audio_dir"] / ".extracted"
        ).is_file():
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "MultiDialog audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        "`metadata.jsonl` and the extracted audio, or let `prepare_source` "
        "download them."
    )


def _hf_files() -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{_CFG['hf_repo']}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["siblings"]


def _fetch(name: str, size: int, destination: Path, attempts: int = 5) -> None:
    """Download one repository file, resuming a partial download."""
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{_CFG['hf_repo']}/resolve/main/{name}"
    for _ in range(attempts):
        have = target.stat().st_size if target.is_file() else 0
        if have == size:
            return
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={have}-"} if have else {}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with target.open("ab" if have else "wb") as fh:
                    shutil.copyfileobj(response, fh, 1 << 20)
        except OSError as err:
            logger.warning("Retrying %s: %s", name, err)
    if target.stat().st_size != size:
        raise ValueError(f"Incomplete download: {name}")


def _download_source(destination: Path) -> None:
    """Fetch the metadata and the audio archives of the Hugging Face dataset.

    Files that already have the listed size are kept.

    Raises:
        ValueError: If a file does not reach its listed size.
    """
    files = [
        f
        for f in _hf_files()
        if f["rfilename"] == _CFG["metadata_file"]
        or (
            f["rfilename"].startswith(f"{_CFG['archive_dir']}/")
            and f["rfilename"].endswith(".tar.gz")
        )
    ]
    for entry in files:
        _fetch(entry["rfilename"], int(entry["size"]), destination)
        logger.info("Downloaded %s", entry["rfilename"])


def _extract_archives(source_root: Path) -> None:
    """Extract ``data/<split>/*.tar.gz`` into ``audio/<split>/``."""
    audio_root = source_root / _CFG["audio_dir"]
    marker = audio_root / ".extracted"
    if marker.is_file():
        return
    for archive in sorted((source_root / _CFG["archive_dir"]).glob("*/*.tar.gz")):
        out = audio_root / archive.parent.name
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(out, filter="data")
        logger.info("Extracted %s", archive.name)
    marker.write_text("done\n")


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$MULTIDIALOG_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _drop_duplicates(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Leave out the utterances listed in ``duplicate_ids`` of the config.

    Applies only when ``exclude_duplicates`` is true. Each listed utterance has
    the same sound and label as one that stays.

    Raises:
        ValueError: If a listed id is not among ``entries``.
    """
    if not _CFG["exclude_duplicates"]:
        return entries
    drop = {str(i) for i in _CFG["duplicate_ids"]}
    kept = [e for e in entries if e["utt_id"] not in drop]
    missing = drop - {e["utt_id"] for e in entries}
    if missing:
        raise ValueError(f"`duplicate_ids` not found: {sorted(missing)}")
    return kept


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the metadata and match each row to its audio file.

    Args:
        source_root: Directory returned by :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker``,
        ``split`` and ``label``.

    Raises:
        ValueError: If a label or split is unknown, there are fewer than
            ``expected_rows`` rows, or an id of ``conflicting_ids`` is absent.
        FileNotFoundError: If an audio file listed in the metadata is missing.
    """
    with (source_root / _CFG["metadata_file"]).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if len(rows) < int(_CFG["expected_rows"]):
        raise ValueError(
            f"metadata has {len(rows)} rows, expected {_CFG['expected_rows']}"
        )

    audio_root = source_root / _CFG["audio_dir"]
    entries = []
    for row in rows:
        if row["emotion"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {row['emotion']!r}. Known: {sorted(_EMOTION_MAP)}"
            )
        split_dir = row["audpath"].split("/", 1)[0]
        if split_dir not in _SPLIT_MAP:
            raise ValueError(f"Unknown split {split_dir!r}")
        wav = audio_root / row["audpath"]
        if not wav.is_file():
            raise FileNotFoundError(f"Audio listed in the metadata is missing: {wav}")
        stem = Path(row["file_name"]).stem
        entries.append(
            {
                "utt_id": f"multidialog-{stem}",
                "source": str(wav),
                "speaker": stem[-1],
                "split": _SPLIT_MAP[split_dir],
                "label": _MERGE_LABELS.get(
                    _EMOTION_MAP[row["emotion"]], _EMOTION_MAP[row["emotion"]]
                ),
            }
        )
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    conflicting = {str(i) for i in _CFG["conflicting_ids"]}
    missing = conflicting - {e["utt_id"] for e in entries}
    if missing:
        raise ValueError(f"`conflicting_ids` not found: {sorted(missing)}")
    entries = [e for e in entries if e["utt_id"] not in conflicting]
    return _drop_duplicates(entries)


class MultidialogBuilder(DatasetBuilder):
    """Prepare MultiDialog for the classification recipe.

    `prepare_source` downloads and extracts the audio archives. `build()`
    converts every utterance to 16 kHz mono and writes one manifest per split:
    train, the two validation sets together, and the two test sets together.
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
        """Download and extract the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("MultiDialog already present; skipping download")
            return
        destination = next(iter_source_candidates(recipe_root, source_dir))
        _download_source(destination)
        _extract_archives(destination)
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
            source_dir: Explicit location of the MultiDialog files.

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
