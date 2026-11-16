"""EmoTa dataset builder (Sri Lankan Tamil acted speech, 5 emotions)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
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
# `01_05_ang.wav` -> speaker 01, sentence 05, emotion code `ang`.
FILENAME_RE = re.compile(str(_CFG["file_re"]))


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the EmoTa source, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate source roots, most specific first.
    """
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def _audio_files(root: Path) -> list[Path]:
    """Return the recordings under ``root`` whose name matches the release's pattern."""
    audio_dir = root / str(_CFG["audio_subdir"])
    if not audio_dir.is_dir():
        return []
    return sorted(p for p in audio_dir.glob("*.wav") if FILENAME_RE.match(p.name))


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every recording of the release.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Source root holding ``data/<speaker>_<sentence>_<emotion>.wav``.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    expected = int(_CFG["expected_files"])
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        found = len(_audio_files(candidate))
        if found >= expected:
            return candidate
        checked.append(f"{candidate} ({found} of {expected} recordings)")
    raise FileNotFoundError(
        "Complete EmoTa corpus not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        f"`{_CFG['audio_subdir']}/*.wav`, or let `prepare_source` download "
        "them from Hugging Face."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMOTA_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _download_source(destination: Path) -> None:
    """Fetch the recordings from Hugging Face at the pinned revision.

    Files already present in ``destination`` are not fetched again, so an
    interrupted download resumes.

    Args:
        destination: Source root to download into.

    Raises:
        PermissionError: If the account has not accepted the dataset's terms
            or no Hugging Face login is available.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s into %s", _CFG["hf_repo"], destination)
    try:
        snapshot_download(
            str(_CFG["hf_repo"]),
            repo_type="dataset",
            revision=str(_CFG["hf_revision"]),
            allow_patterns=[str(p) for p in _CFG["allow_patterns"]],
            local_dir=destination,
        )
    except GatedRepoError as err:
        raise PermissionError(
            f"{_CFG['hf_repo']} is a gated dataset and this Hugging Face account "
            "has no access to it. Open "
            f"{_CFG['hf_page']} while logged in, fill in the form and accept "
            "the license terms (access is granted automatically), then log in "
            "on this machine with `huggingface-cli login` (or set `HF_TOKEN`) "
            "and run `prepare_source` again. Alternatively, set "
            f"${_CFG['source_env_var']} to a copy you already have."
        ) from err


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the file name.

    Args:
        source_root: Source root (see :func:`resolve_source_root`).

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a file's emotion code is not in ``emotion_map``.
        FileNotFoundError: If no recording is found.
    """
    audio_dir = source_root / str(_CFG["audio_subdir"])
    skipped = sum(
        1 for p in audio_dir.glob("*.wav") if FILENAME_RE.match(p.name) is None
    )
    entries = []
    for wav in _audio_files(source_root):
        fields = FILENAME_RE.match(wav.name)
        if fields["emotion"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion code {fields['emotion']!r} in {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        entries.append(
            {
                "utt_id": f"emota-{wav.stem}",
                "source": str(wav),
                "speaker": fields["speaker"],
                "label": _EMOTION_MAP[fields["emotion"]],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if not entries:
        raise FileNotFoundError(f"No EmoTa audio under {audio_dir}")
    logger.info(
        "Read %d utterance(s) from %d speaker(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. Speakers are not held out,
    so the splits share speakers.

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


def _convert(ffmpeg: str, source: str, wav: Path, sampling_rate: str) -> None:
    """Write one recording as 16 kHz mono 16-bit PCM."""
    part = wav.with_suffix(wav.suffix + ".part")
    subprocess.run(
        [ffmpeg, "-i", source, "-ac", "1", "-ar", sampling_rate]
        + ["-c:a", "pcm_s16le", "-f", "wav", "-y", "-hide_banner"]
        + ["-loglevel", "error", str(part)],
        check=True,
    )
    part.replace(wav)


class EmoTaBuilder(DatasetBuilder):
    """Prepare EmoTa for the classification recipe.

    The recordings are 48 kHz WAV, so `build()` converts every one to 16 kHz
    mono and writes one manifest per split, drawn as a label-stratified
    random split (see :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete EmoTa source is reachable."""
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
        """Download the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            PermissionError: If the gated dataset cannot be accessed.
            FileNotFoundError: If the source is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("EmoTa already present; skipping download")
            return

        destination = next(iter_source_candidates(recipe_root, source_dir))
        _download_source(destination)
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
        num_workers: int = 8,
        **_kwargs,
    ) -> None:
        """Resample the recordings and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location.
            num_workers: ffmpeg processes run at the same time.

        Raises:
            FileNotFoundError: If the source or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a conversion fails.
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

        todo = [
            (e["source"], wav_dir / f"{e['utt_id']}.wav")
            for e in entries
            if not (wav_dir / f"{e['utt_id']}.wav").is_file()
        ]
        logger.info("Converting %d of %d recording(s)", len(todo), len(entries))
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            list(pool.map(lambda t: _convert(ffmpeg, t[0], t[1], sampling_rate), todo))

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in entries:
            wav = wav_dir / f"{entry['utt_id']}.wav"
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
