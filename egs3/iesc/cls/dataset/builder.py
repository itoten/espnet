"""IESC dataset builder (Kaggle manual download, speaker-held-out split).

Walks `Speaker-<n>/<Emotion>/*.wav`, takes the label from the emotion directory
and resamples every recording to 16 kHz mono into `$IESC_OUTPUT/wav`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
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
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the corpus tree, in order."""
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def _list_wavs(root: Path) -> list[Path]:
    return sorted(root.glob(str(_CFG["audio_glob"])))


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding all ``expected_files`` recordings.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir():
            n_found = len(_list_wavs(candidate))
            if n_found >= int(_CFG["expected_files"]):
                return candidate
            checked.append(f"{candidate} ({n_found} recordings)")
        else:
            checked.append(f"{candidate} (not a directory)")
    raise FileNotFoundError(
        "IESC audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + "\n\nThe Kaggle download needs a login, so this recipe cannot fetch it. "
        f"Download\n  {_CFG['request_url']}\n"
        "extract the zip, and set "
        f"${_CFG['source_env_var']} to the directory that holds the "
        "`Speaker-<n>/` folders (or a parent of it)."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory holding the manifests, token list and WAV files.

    Returns:
        Path: ``$IESC_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / str(_CFG["data_path"])


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking label and speaker from the directories.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If an emotion directory is unknown, a name repeats, or an
            `exclude_files` path is absent.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: set[str] = set()
    excluded = set(_CFG["exclude_files"])
    for wav in _list_wavs(source_root):
        emotion, speaker = wav.parent.name, wav.parent.parent.name
        rel = f"{speaker}/{emotion}/{wav.name}"
        if rel in excluded:
            excluded.discard(rel)
            continue
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion directory {emotion!r} for {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        utt_id = f"iesc-{speaker}-{wav.stem}"
        if utt_id in seen:
            raise ValueError(f"Duplicate utterance id {utt_id}")
        seen.add(utt_id)
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "speaker": speaker,
                "label": _EMOTION_MAP[emotion],
            }
        )
    if excluded:
        raise ValueError(f"`exclude_files` paths not found: {sorted(excluded)}")
    if not entries:
        raise FileNotFoundError(f"No IESC audio under {source_root}")
    logger.info("Read %d recording(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole speakers: ``n_test_speakers``, ``n_valid_speakers``, rest train.

    Speakers are shuffled with ``split_seed``, so every split has speakers the
    others never see.

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
    return {e["utt_id"]: split_of.get(e["speaker"], "train") for e in entries}


class IescBuilder(DatasetBuilder):
    """Prepare IESC for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one manifest
    per split, drawn as a speaker-held-out split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete source tree is reachable."""
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
        """Check that the corpus has been placed by hand.

        Raises:
            FileNotFoundError: With the download instructions if it is absent.
        """
        resolve_source_root(Path(recipe_dir).resolve(), source_dir)

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
