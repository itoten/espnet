"""MSP-Podcast dataset builder."""

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
    """Yield the directories that may hold the corpus tree, in order.

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


def missing_source_entries(root: Path) -> list[str]:
    """Return the corpus entries that are absent under ``root``.

    Args:
        root: Candidate corpus root.

    Returns:
        list[str]: Paths that a complete source tree would contain.
    """
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

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root containing ``Audios/`` and ``Labels/``.

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
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$MSPPODCAST_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Read the consensus table into one row per usable recording.

    Rows whose partition is not in ``split_map`` are skipped: Test3 has no
    labels of its own, since they are withheld for the challenge's submission
    interface.

    Args:
        source_root: Corpus root.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``label`` and
        ``split``.

    Raises:
        ValueError: If a row carries an emotion code that is neither kept nor
            listed for dropping.
        FileNotFoundError: If a recording named in the table is absent.
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
            "Dropped %d recording(s) by emotion code (%s); see " "dataset/config.yaml",
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
    """Prepare MSP-Podcast for the classification recipe.

    The corpus needs a licence agreement, so it is placed by hand; `build()`
    resamples every recording to 16 kHz mono and writes one manifest per split.
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
        """Check that a hand-placed corpus is there; never download.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the corpus is not where it is expected, with
                the request page and the layout in the message.
        """
        resolve_source_root(Path(recipe_dir).resolve(), source_dir)
        logger.info("MSP-Podcast found; nothing to download")

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
        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for done, entry in enumerate(entries, 1):
            wav = wav_dir / f"{entry['utt_id']}.wav"
            if not wav.is_file():
                # `-f wav` sets the format, so the `.part` extension is fine.
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
            rows[entry["split"]].append(
                (entry["utt_id"], str(wav.resolve()), entry["label"])
            )
            # The corpus runs to 200k recordings, so progress is worth logging.
            if done % 20000 == 0:
                logger.info("Converted %d/%d", done, len(entries))

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
