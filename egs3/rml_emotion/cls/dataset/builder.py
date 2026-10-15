"""RML Emotion Database builder (manually extracted Kaggle data)."""

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

# `s4/s4/f2chi/an1.avi`: subject s4, language folder f2chi, emotion code an,
# take 1. Subjects s1 and s2 have no language folder.
PATH_RE = re.compile(
    r"^(?P<speaker>s\d+)/(?P=speaker)/(?:(?P<lang>f\d[a-z]+)/)?"
    r"(?P<emotion>[a-z]+)(?P<take>\d+)\.avi$"
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
    """Yield the directories that may hold the RML source tree, in order.

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


def _iter_avi(root: Path) -> Iterator[tuple[Path, re.Match]]:
    for avi in sorted(root.glob("s*/s*/**/*.avi")):
        match = PATH_RE.match(avi.relative_to(root).as_posix())
        if match:
            yield avi, match


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every recording.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding the ``s1/`` ... ``s8/`` folders.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and sum(1 for _ in _iter_avi(candidate)) >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "RML Emotion Database not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nDownload and unzip {_CFG['kaggle_url']} (free Kaggle account), then "
        f"set ${_CFG['source_env_var']} to the directory holding the "
        "`s1/` ... `s8/` folders."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$RML_EMOTION_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the file name.

    Args:
        source_root: Corpus root from :func:`resolve_source_root`.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``group`` (the
        subject folder) and ``label``.

    Raises:
        ValueError: If a file's emotion code is unknown.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    for avi, fields in _iter_avi(source_root):
        if fields["emotion"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion code {fields['emotion']!r} in {avi}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        parts = [fields["speaker"], fields["lang"], fields["emotion"] + fields["take"]]
        entries.append(
            {
                "utt_id": "rml_emotion-" + "-".join(p for p in parts if p),
                "source": str(avi),
                "group": fields["speaker"],
                "label": _EMOTION_MAP[fields["emotion"]],
            }
        )
    if not entries:
        raise FileNotFoundError(f"No RML recordings under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d subject(s)",
        len(entries),
        len({e["group"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split whole subjects at random so no subject spans two splits.

    Subjects are shuffled with ``split_seed`` and filled into test, then valid,
    then train until each holds about its ratio of the utterances. Labels are
    not balanced across splits.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If the three ratios do not sum to 1.
    """
    train_ratio = float(_CFG["train_ratio"])
    valid_ratio = float(_CFG["valid_ratio"])
    test_ratio = float(_CFG["test_ratio"])
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            "`train_ratio`/`valid_ratio`/`test_ratio` must sum to 1, got "
            f"{train_ratio} + {valid_ratio} + {test_ratio}"
        )

    by_group: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        by_group.setdefault(entry["group"], []).append(entry["utt_id"])
    groups = sorted(by_group)
    Random(int(_CFG["split_seed"])).shuffle(groups)

    total = len(entries)
    quota = {"test": test_ratio * total, "valid": valid_ratio * total}
    counts = {"test": 0, "valid": 0}
    assignment: dict[str, str] = {}
    for group in groups:
        for split in ("test", "valid"):
            if counts[split] < quota[split]:
                break
        else:
            split = "train"
        if split != "train":
            counts[split] += len(by_group[group])
        for utt_id in by_group[group]:
            assignment[utt_id] = split
    return assignment


class RmlEmotionBuilder(DatasetBuilder):
    """Prepare the RML Emotion Database for the classification recipe.

    `build()` extracts the audio track of every avi as 16 kHz mono wav and
    writes one manifest per split, drawn as a random split of whole subjects
    (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether an RML source tree is reachable."""
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
        """Check that the manually extracted corpus is available.

        Raises:
            FileNotFoundError: With the Kaggle URL if the tree is missing.
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
        """Extract the audio and write one TSV manifest per split.

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
            raise FileNotFoundError(
                "ffmpeg is required to extract 16 kHz mono audio from the avi files"
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
