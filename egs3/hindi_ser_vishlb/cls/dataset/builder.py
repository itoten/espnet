"""Hindi SER (vishlb) dataset builder (manually obtained Kaggle archive)."""

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


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the extracted archive, in order.

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
    """Return the first candidate holding the full set of recordings.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory holding at least ``expected_files`` recordings below it.

    Raises:
        FileNotFoundError: If no candidate qualifies.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and len(list(_iter_wavs(candidate))) >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete Hindi SER (vishlb) audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nDownload the archive from {_CFG['kaggle_url']} (a free Kaggle "
        "account is needed), extract it, and set "
        f"${_CFG['source_env_var']} to the extracted directory."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$HINDI_SER_VISHLB_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


# `<speaker>/session<n>/<emotion>/1.5.surprise-01.wav`: speaker 1, session 5,
# sentence 01. The emotion comes from the folder, because some names have typos
# or spaces (`4.3.disgust 07.wav`, `2.5.dusgust-09.wav`) and one file sits in
# the wrong session folder (`4.5.happy-08.wav` under `session4`).
FILENAME_RE = re.compile(
    r"^(?P<speaker>\d+)\.(?P<session>\d+)\.\s*[A-Za-z]+[-\s]+(?P<sentence>\d+)\.wav$"
)


def _iter_wavs(root: Path) -> Iterator[Path]:
    return iter(sorted(root.rglob("*.wav")))


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
    """Collect the recordings: label from the folder, speaker and session from the name.

    Args:
        source_root: Directory holding the ``<speaker>/session<n>/<emotion>``
            folders somewhere below it.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``group`` (the
        speaker) and ``label``.

    Raises:
        ValueError: If a name does not parse, its speaker differs from the
            folder's, a folder's emotion is unknown, or two files map to the same
            utterance id.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: dict[str, Path] = {}
    for wav in _iter_wavs(source_root):
        speaker, _, emotion = wav.parts[-4:-1]
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            raise ValueError(f"Unexpected file name: {wav}")
        if fields["speaker"] != speaker:
            raise ValueError(f"Speaker in the name and the folder disagree: {wav}")
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion folder {emotion!r} for {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        utt_id = (
            f"hindi_ser_vishlb-{speaker}-{fields['session']}-{emotion}-"
            f"{fields['sentence']}"
        )
        if utt_id in seen:
            raise ValueError(f"Duplicate utterance id: {wav} and {seen[utt_id]}")
        seen[utt_id] = wav
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "group": speaker,
                "label": _EMOTION_MAP[emotion],
            }
        )
    if not entries:
        raise FileNotFoundError(f"No Hindi SER (vishlb) audio under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["group"] for e in entries}),
    )
    return _drop_duplicates(entries)


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole speakers: ``n_test_groups``, ``n_valid_groups``, rest train.

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
    n_test = int(_CFG["n_test_groups"])
    n_valid = int(_CFG["n_valid_groups"])
    groups = sorted({e["group"] for e in entries})
    if len(groups) < n_test + n_valid + 1:
        raise ValueError(f"Only {len(groups)} speakers for {n_test}+{n_valid}+train")
    Random(int(_CFG["split_seed"])).shuffle(groups)
    split_of = {g: "test" for g in groups[:n_test]}
    split_of.update({g: "valid" for g in groups[n_test : n_test + n_valid]})
    return {
        e["utt_id"]: split_of.get(e["group"], "train")
        for e in sorted(entries, key=lambda e: e["utt_id"])
    }


class HindiSerVishlbBuilder(DatasetBuilder):
    """Prepare Hindi SER (vishlb) for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes
    one manifest per split, drawn as a speaker-held-out split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a Hindi SER (vishlb) source tree is reachable."""
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
        """Check that the manually downloaded corpus is in place.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus is missing; the message says where
                to download it.
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
