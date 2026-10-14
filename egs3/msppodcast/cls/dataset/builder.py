"""MSP-Podcast dataset builder (8 classes, Train and Development).

Reads `labels_consensus.csv` and the `Audios/` tree, keeps the eight emotion
classes including `contempt` (see `dataset/config.yaml`), and drops `O`/`X`.
Only the `Train`/`Development` partitions feed training and validation; the
`Test1`/`Test2` manifests are built too so the model can be scored on them.
Audio is resampled to 16 kHz mono into `$MSPPODCAST_OUTPUT/wav`.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
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
_EMOTION_CODES = {str(k): str(v) for k, v in _CFG["emotion_codes"].items()}
_DROP_CODES = {str(c) for c in _CFG["drop_codes"]}
_SPLIT_MAP = {str(k): str(v) for k, v in _CFG["split_map"].items()}


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


def missing_source_entries(root: Path) -> list[str]:
    """Return the corpus entries that are absent under ``root``."""
    audio_dir = root / str(_CFG["audio_subdir"])
    missing = []
    if not (audio_dir.is_dir() and any(audio_dir.glob("MSP-PODCAST_*.wav"))):
        missing.append(str(audio_dir))
    labels = root / str(_CFG["labels_csv"])
    if not labels.is_file():
        missing.append(str(labels))
    return missing


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding both the audio and the label table.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        missing = missing_source_entries(candidate)
        if not missing:
            return candidate
        checked.append(f"{candidate} (missing: {', '.join(missing)})")
    raise FileNotFoundError(
        "MSP-Podcast source tree not found. Checked:\n  "
        + "\n  ".join(checked)
        + "\n\nThe corpus is released under a licence agreement and cannot be "
        f"downloaded by this recipe. Request it from\n  {_CFG['request_url']}\n"
        f"and set ${_CFG['source_env_var']} to a directory holding:\n"
        f"  {_CFG['audio_subdir']}/MSP-PODCAST_*.wav\n"
        f"  {_CFG['labels_csv']}"
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory holding the manifests, token list and WAV files.

    Returns:
        Path: ``$MSPPODCAST_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / str(_CFG["data_path"])


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the consensus table into one row per usable recording.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``label`` and
        ``split``.

    Raises:
        ValueError: If a row carries an emotion code that is neither kept nor
            listed for dropping.
        FileNotFoundError: If more than ``max_missing_audio`` recordings named
            in the table are absent.
    """
    audio_dir = source_root / str(_CFG["audio_subdir"])
    wanted = {v: k for k, v in _SPLIT_MAP.items()}

    entries = []
    dropped: dict[str, int] = {}
    n_other_split = 0
    with (source_root / str(_CFG["labels_csv"])).open(
        "r", encoding="utf-8", newline=""
    ) as fh:
        for row in csv.DictReader(fh):
            split = wanted.get(row["Split_Set"].strip())
            if split is None:
                n_other_split += 1
                continue
            code = row["EmoClass"].strip()
            if code in _DROP_CODES:
                dropped[code] = dropped.get(code, 0) + 1
                continue
            if code not in _EMOTION_CODES:
                raise ValueError(
                    f"Unknown emotion code {code!r} for {row['FileName']}. "
                    f"Known: {sorted(_EMOTION_CODES)}; "
                    f"dropped: {sorted(_DROP_CODES)}"
                )
            name = row["FileName"].strip()
            entries.append(
                {
                    "utt_id": f"msppodcast-{Path(name).stem}",
                    "source": str(audio_dir / name),
                    "label": _EMOTION_CODES[code],
                    "split": split,
                }
            )

    if dropped:
        logger.info(
            "Dropped %d recording(s) by emotion code (%s); see dataset/config.yaml",
            sum(dropped.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())),
        )
    if n_other_split:
        logger.info("Skipped %d row(s) outside the configured splits", n_other_split)

    missing = [e for e in entries if not Path(e["source"]).is_file()]
    allowed = int(_CFG["max_missing_audio"])
    if len(missing) > allowed:
        raise FileNotFoundError(
            f"{len(missing)} recording(s) named in the table are absent under "
            f"{audio_dir}, starting with {Path(missing[0]['source']).name}. "
            f"`max_missing_audio` allows {allowed}, so this copy looks "
            "incomplete rather than merely inconsistent."
        )
    if missing:
        gone = {e["utt_id"] for e in missing}
        logger.warning(
            "Skipped %d recording(s) the release does not ship: %s",
            len(missing),
            ", ".join(sorted(Path(e["source"]).name for e in missing)),
        )
        entries = [e for e in entries if e["utt_id"] not in gone]
    logger.info("Read %d recording(s)", len(entries))
    return entries


class MSPPodcastBuilder(DatasetBuilder):
    """Prepare MSP-Podcast (8 classes) for this recipe."""

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
        """Check that the hand-placed corpus is there; never download."""
        resolve_source_root(Path(recipe_dir).resolve(), source_dir)
        logger.info("MSP-Podcast found; nothing to download")

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether this recipe's own manifests already exist."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        paths = [str(p) for p in _CFG["manifest_paths"].values()]
        return all((data_root / p).is_file() for p in paths)

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Resample the recordings and write the split manifests.

        Raises:
            FileNotFoundError: If the corpus or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a resample fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)
        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required to resample the recordings to 16 kHz"
            )
        sampling_rate = str(_CFG["sampling_rate"])

        entries = read_entries(source_root)
        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        n_converted = 0
        for done, entry in enumerate(entries, 1):
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
                n_converted += 1
            rows[entry["split"]].append(
                (entry["utt_id"], str(wav.resolve()), entry["label"])
            )
            if done % 20000 == 0:
                logger.info(
                    "Checked %d/%d, converted %d new", done, len(entries), n_converted
                )
        logger.info("Converted %d new recording(s); reused the rest", n_converted)

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
