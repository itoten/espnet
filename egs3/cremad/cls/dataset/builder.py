"""CREMA-D dataset builder."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from importlib import resources
from pathlib import Path
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# `1001_DFA_ANG_XX.wav` -> actor 1001, sentence DFA, emotion ANG, intensity XX.
FILENAME_RE = re.compile(
    r"^(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>[A-Z]{3})"
    r"_(?P<intensity>[A-Z]{1,2})\.wav$"
)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_CODES = {str(k): str(v) for k, v in _CFG["emotion_codes"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the CREMA-D source tree, in order.

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


def resolve_audio_dir(root: Path) -> Path | None:
    """Return the directory of WAV files under ``root``, or None.

    Accepts either the repository layout (``<root>/AudioWAV``) or a directory
    of WAV files handed over directly, so a corpus copy that was flattened
    elsewhere still works.

    Args:
        root: Candidate corpus root.

    Returns:
        Path | None: Directory containing the WAV files, or None when neither
        layout matches.
    """
    nested = root / str(_CFG["audio_subdir"])
    for candidate in (nested, root):
        if candidate.is_dir() and any(candidate.glob("*.wav")):
            return candidate
    return None


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate that holds CREMA-D audio.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory containing the WAV files.

    Raises:
        FileNotFoundError: If no candidate holds any WAV file.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        audio_dir = resolve_audio_dir(candidate)
        if audio_dir is not None:
            return audio_dir
        checked.append(str(candidate))
    raise FileNotFoundError(
        "CREMA-D audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a corpus root, or let "
        "`prepare_source` download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$CREMAD_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _require_git_lfs() -> None:
    """Fail early when git-lfs is missing.

    The corpus tracks every ``*.wav`` through Git LFS, so a clone without the
    extension installed produces pointer files that still carry the ``.wav``
    name. Nothing downstream would notice until the audio is read.

    Raises:
        RuntimeError: If ``git lfs`` is unavailable.
    """
    try:
        subprocess.run(
            ["git", "lfs", "version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "git-lfs is required to download CREMA-D: the corpus stores its "
            "audio in Git LFS, and a clone without it yields pointer files "
            "named *.wav. Install it (https://git-lfs.com) and retry, or set "
            f"${_CFG['source_env_var']} to a corpus copy you already have."
        ) from exc


def _assert_real_audio(audio_dir: Path) -> None:
    """Check that the WAV files are audio rather than Git LFS pointers.

    Args:
        audio_dir: Directory holding the corpus audio.

    Raises:
        RuntimeError: If the first file is not RIFF-headed.
    """
    sample = next(iter(sorted(audio_dir.glob("*.wav"))), None)
    if sample is None:
        return
    with sample.open("rb") as fh:
        header = fh.read(4)
    if header != b"RIFF":
        raise RuntimeError(
            f"{sample} is not a WAV file (header {header!r}). Git LFS content "
            "was probably not fetched; run `git lfs pull` inside "
            f"{audio_dir.parent}."
        )


def _clone_repository(destination: Path) -> None:
    """Shallow-clone the CREMA-D repository into ``destination``.

    Args:
        destination: Directory to clone into.

    Raises:
        RuntimeError: If git-lfs is unavailable.
        subprocess.CalledProcessError: If the clone fails.
    """
    _require_git_lfs()
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = str(_CFG["repo_url"])
    logger.info("Cloning CREMA-D (about 600 MB of LFS audio) from %s", url)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)],
        check=True,
    )


def parse_filename(name: str) -> dict[str, str] | None:
    """Split a CREMA-D file name into its fields.

    Args:
        name: File name such as ``1001_DFA_ANG_XX.wav``.

    Returns:
        dict[str, str] | None: Keys ``actor``, ``sentence``, ``emotion`` and
        ``intensity``, or None when the name does not match the corpus scheme.

    Examples:
        >>> parse_filename("1001_DFA_ANG_XX.wav")["emotion"]
        'ANG'
    """
    match = FILENAME_RE.match(name)
    return match.groupdict() if match else None


def assign_splits(speakers: list[str]) -> dict[str, str]:
    """Deal sorted speaker ids out to the configured splits.

    CREMA-D has no official partition. Sorting first makes the assignment a
    function of the corpus alone, so the split is reproducible without shipping
    a list of ids. The counts must account for every speaker: a partial
    assignment is far more likely to be a typo in the config than a deliberate
    choice to drop actors, so it is rejected rather than patched up.

    TODO(itoten): Take the speaker ids from the config instead of deriving them
    from sort order, so the split can reproduce a published partition. Sorted
    ids ignore the gender balance that a hand-built fold can hold.

    Args:
        speakers: Speaker ids found in the corpus.

    Returns:
        dict[str, str]: Speaker id to split name.
        ex) {"1001": "train", "1002": "valid", "1003": "test"}

    Raises:
        ValueError: If the configured counts do not sum to the number of
            speakers in the corpus.
    """
    counts = {str(k): int(v) for k, v in _CFG["split_speaker_counts"].items()}
    ordered = sorted(set(speakers))
    if sum(counts.values()) != len(ordered):
        raise ValueError(
            f"`split_speaker_counts` sums to {sum(counts.values())} but the "
            f"corpus has {len(ordered)} speakers "
            f"({', '.join(f'{k}={v}' for k, v in counts.items())}). "
            "Adjust the counts in dataset/config.yaml so every speaker lands "
            "in exactly one split."
        )

    assignment = {}
    start = 0
    for split, count in counts.items():
        for speaker in ordered[start : start + count]:
            assignment[speaker] = split
        start += count
    return assignment


class CremaDBuilder(DatasetBuilder):
    """Prepare CREMA-D for the classification recipe.

    The corpus is already distributed as 16 kHz mono WAV, so `build()` only
    writes one manifest per split; the audio is read from wherever it was
    downloaded or placed.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a directory of CREMA-D WAV files is reachable."""
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
        """Clone the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            subprocess.CalledProcessError: If the clone fails.
            FileNotFoundError: If no audio is present after cloning.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("CREMA-D already present; skipping download")
            return

        destination = next(iter_source_candidates(recipe_root, source_dir))
        _clone_repository(destination)
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
        """Write one TSV manifest per split.

        Each row is ``utterance_id<TAB>wav_path<TAB>label``. Utterance ids are
        the file stems, which already encode speaker, sentence, emotion and
        intensity.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus audio cannot be found.
            ValueError: If the corpus holds too few speakers to split.
        """
        recipe_root = Path(recipe_dir).resolve()
        audio_dir = resolve_source_root(recipe_root, source_dir)
        _assert_real_audio(audio_dir)
        data_root = resolve_data_root(recipe_root)

        entries = []
        skipped = 0
        for wav in sorted(audio_dir.glob("*.wav")):
            fields = parse_filename(wav.name)
            if fields is None or fields["emotion"] not in _EMOTION_CODES:
                skipped += 1
                continue
            entries.append(
                (
                    fields["actor"],
                    wav.stem,
                    str(wav.resolve()),
                    _EMOTION_CODES[fields["emotion"]],
                )
            )
        if skipped:
            logger.warning("Skipped %d file(s) with an unexpected name", skipped)
        if not entries:
            raise FileNotFoundError(f"No CREMA-D audio under {audio_dir}")

        assignment = assign_splits([actor for actor, _, _, _ in entries])
        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                (utt_id, wav, label)
                for actor, utt_id, wav, label in entries
                if assignment[actor] == split
            ]
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            speakers = len({r[0].split("_")[0] for r in rows})
            logger.info(
                "%s: wrote %d entries from %d speaker(s) to %s",
                split,
                len(rows),
                speakers,
                manifest,
            )
