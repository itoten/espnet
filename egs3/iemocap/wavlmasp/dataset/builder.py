"""IEMOCAP dataset builder."""

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

SESSION_DIR_RE = re.compile(r"^Session(?P<number>[1-5])$")

# "[6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]";
# the per-annotator lines after it ("C-E2:", "A-E3:") do not match.
EMOEVAL_LINE_RE = re.compile(
    r"^\[[\d.]+\s*-\s*[\d.]+\]\t(?P<turn_id>\S+)\t(?P<label>[a-z]{3})\t"
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
    """Yield the directories that may hold the IEMOCAP source tree, in order.

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


def iter_session_dirs(root: Path) -> Iterator[Path]:
    """Yield every ``SessionN`` directory under ``root``, in session order.

    Args:
        root: Candidate corpus root.

    Yields:
        Path: ``SessionN`` directories that exist.
    """
    if not root.is_dir():
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and SESSION_DIR_RE.match(entry.name):
            yield entry


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding at least one ``SessionN`` directory.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding one or more ``SessionN/`` dirs.

    Raises:
        FileNotFoundError: If no candidate holds any.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if any(True for _ in iter_session_dirs(candidate)):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "IEMOCAP not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root holding "
        "`Session1`..`Session5`; this corpus needs a signed release "
        "agreement (https://sail.usc.edu/iemocap/iemocap_release.htm) and "
        "cannot be fetched from a script."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$IEMOCAP_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _read_emo_evaluation(session_dir: Path) -> dict[str, str]:
    """Read every turn's categorical label from one session's evaluations.

    Args:
        session_dir: A ``SessionN`` directory.

    Returns:
        dict[str, str]: Turn id (e.g. ``Ses01F_impro01_F000``) to its raw
        3-letter label code (e.g. ``neu``), for every turn the session's
        ``EmoEvaluation/*.txt`` files carry -- including labels outside
        ``emotion_map``, so callers can tell "no label" apart from "label
        not selected for this recipe".
    """
    eval_dir = session_dir / "dialog" / "EmoEvaluation"
    labels: dict[str, str] = {}
    if not eval_dir.is_dir():
        return labels
    for txt in sorted(eval_dir.glob("*.txt")):
        with txt.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                match = EMOEVAL_LINE_RE.match(line)
                if match is None:
                    continue
                labels[match["turn_id"]] = match["label"]
    return labels


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings whose label is in ``emotion_map``.

    Args:
        source_root: Corpus root holding the ``SessionN/`` directories.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``. ``speaker`` is the turn id's own session+gender prefix
        (e.g. ``Ses01F``); it is not read for splitting (see
        :func:`assign_splits`), only kept for traceability.

    Raises:
        FileNotFoundError: If no recording is found.
    """
    entries = []
    for session_dir in iter_session_dirs(source_root):
        raw_labels = _read_emo_evaluation(session_dir)
        wav_root = session_dir / "sentences" / "wav"
        if not wav_root.is_dir():
            continue
        seen_turn_ids: set[str] = set()
        for dialog_dir in sorted(wav_root.iterdir()):
            if not dialog_dir.is_dir():
                continue
            for wav in sorted(dialog_dir.glob("*.wav")):
                turn_id = wav.stem
                seen_turn_ids.add(turn_id)
                raw_label = raw_labels.get(turn_id)
                if raw_label is None or raw_label not in _EMOTION_MAP:
                    continue
                speaker_match = re.match(r"^(?P<speaker>Ses\d{2}[MF])_", turn_id)
                entries.append(
                    {
                        "utt_id": f"iemocap-{turn_id}",
                        "source": str(wav),
                        "speaker": (
                            speaker_match["speaker"] if speaker_match else "unknown"
                        ),
                        "label": _EMOTION_MAP[raw_label],
                    }
                )
        # A labeled turn without a WAV is reported rather than treated as an error.
        missing = [
            turn_id
            for turn_id, label in raw_labels.items()
            if label in _EMOTION_MAP and turn_id not in seen_turn_ids
        ]
        if missing:
            logger.warning(
                "%s: %d labeled turn(s) have no matching WAV, e.g. %s",
                session_dir.name,
                len(missing),
                missing[0],
            )
    if not entries:
        raise FileNotFoundError(
            f"No IEMOCAP audio with a mapped label under {source_root}"
        )
    logger.info(
        "Read %d utterance(s) from %d speaker(s) across %d session(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
        len(list(iter_session_dirs(source_root))),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. Sessions and speakers are not held
    out, so the splits share speakers.

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

    rng = Random(int(_CFG["split_seed"]))
    by_label: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        by_label.setdefault(entry["label"], []).append(entry["utt_id"])

    assignment: dict[str, str] = {}
    for _, utt_ids in sorted(by_label.items()):
        shuffled = list(utt_ids)
        rng.shuffle(shuffled)
        n_valid = round(len(shuffled) * valid_ratio)
        n_test = round(len(shuffled) * test_ratio)
        for utt_id in shuffled[:n_test]:
            assignment[utt_id] = "test"
        for utt_id in shuffled[n_test : n_test + n_valid]:
            assignment[utt_id] = "valid"
        for utt_id in shuffled[n_test + n_valid :]:
            assignment[utt_id] = "train"
    return assignment


class IEMOCAPBuilder(DatasetBuilder):
    """Prepare IEMOCAP for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes one
    manifest per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether an IEMOCAP source tree is reachable."""
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
        """Confirm the hand-placed release is present; this cannot be downloaded.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is not present.
        """
        recipe_root = Path(recipe_dir).resolve()
        # No download step exists: `resolve_source_root` raises its own
        # actionable message (pointing at the release agreement) when the
        # tree is missing, so this only needs to surface that.
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
