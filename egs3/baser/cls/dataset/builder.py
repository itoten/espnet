"""BASER dataset builder (5 classes, speaker-held-out split).

Takes the label and speaker from the file names and resamples every recording
to 16 kHz mono into `$BASER_OUTPUT/wav`.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# `M_4_RIAD_S_2_ANGRY_3.wav`: male speaker 4 (RIAD), sentence 2, angry, take 3.
# A few names have no take number, and one speaker name is in mixed case.
NAME_RE = re.compile(
    r"^(?P<sex>[FM])_(?P<num>\d+)_(?P<name>[A-Za-z]+)_S_(?P<sentence>\d+)"
    r"_(?P<emotion>[A-Z]+)_(?P<take>\d*)\.wav$"
)


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


def _count_wavs(root: Path) -> int:
    audio_dir = root / str(_CFG["audio_subdir"])
    return sum(1 for _ in audio_dir.glob("*.wav")) if audio_dir.is_dir() else 0


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the complete audio directory.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        n = _count_wavs(candidate)
        if n >= int(_CFG["expected_files"]):
            return candidate
        checked.append(f"{candidate} ({n} wav files)")
    raise FileNotFoundError(
        "BASER audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + "\n\nThe Kaggle download needs a login, so this recipe cannot fetch it. "
        f"Download\n  {_CFG['request_url']}\n"
        f"and set ${_CFG['source_env_var']} to a directory holding "
        f"{_CFG['audio_subdir']}/ with {_CFG['expected_files']} wav files."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory holding the manifests, token list and WAV files.

    Returns:
        Path: ``$BASER_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / str(_CFG["data_path"])


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking label and speaker from the file name.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker``, ``sex``
        and ``label``.

    Raises:
        ValueError: If a name does not match, an emotion is unknown, two
            files map to the same utterance id, or an `duplicate_files` name is
            absent.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: set[str] = set()
    excluded = (
        set(_CFG["duplicate_files"]) if _CFG["exclude_duplicates"] else set()
    )
    for wav in sorted((source_root / str(_CFG["audio_subdir"])).glob("*.wav")):
        if wav.name in excluded:
            excluded.discard(wav.name)
            continue
        fields = NAME_RE.match(wav.name)
        if fields is None:
            raise ValueError(f"Unexpected recording name {wav.name!r}")
        if fields["emotion"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion {fields['emotion']!r} in {wav.name}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        utt_id = f"baser-{wav.stem.rstrip('_')}"
        if utt_id in seen:
            raise ValueError(f"Duplicate utterance id {utt_id}")
        seen.add(utt_id)
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "speaker": f"{fields['sex']}{fields['num']}",
                "sex": fields["sex"],
                "label": _EMOTION_MAP[fields["emotion"]],
            }
        )
    if excluded:
        raise ValueError(f"`duplicate_files` names not found: {sorted(excluded)}")
    if not entries:
        raise FileNotFoundError(f"No BASER audio under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole speakers: one female and one male for test, one for valid.

    Speakers are shuffled with ``split_seed``; the valid speaker is drawn from
    those left after the test speakers.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If a sex has no speaker for test or too few speakers remain.
    """
    rng = Random(int(_CFG["split_seed"]))
    sex_of = {e["speaker"]: e["sex"] for e in entries}
    test = []
    for sex in ("F", "M"):
        pool = sorted(s for s, x in sex_of.items() if x == sex)
        if not pool:
            raise ValueError(f"No speaker of sex {sex} for the test split")
        test.append(rng.choice(pool))
    rest = sorted(set(sex_of) - set(test))
    if len(rest) < 2:
        raise ValueError(f"Only {len(sex_of)} speakers for test (2) + valid + train")
    valid = rng.choice(rest)
    split_of = {s: "test" for s in test}
    split_of[valid] = "valid"
    return {e["utt_id"]: split_of.get(e["speaker"], "train") for e in entries}


class BaserBuilder(DatasetBuilder):
    """Prepare BASER for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one manifest per
    split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the audio directory is reachable."""
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
