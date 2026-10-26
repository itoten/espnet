"""EmoGator dataset builder (GitHub tarball of mp3 files, contributor split)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# `000001-01-1.mp3`: contributor 000001, category 01, instance 1.
FILENAME_RE = re.compile(
    r"^(?P<speaker>\d{6})-(?P<emotion>\d{2})-(?P<instance>[123])\.mp3$"
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
    """Yield the directories that may hold the EmoGator mp3 files, in order.

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
    """Return the first candidate holding every EmoGator recording.

    Recordings are searched recursively, since they sit in a `data/mp3/`
    directory below the repository root.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root holding all ``<contributor>-<category>-<instance>.mp3``
        files below it.

    Raises:
        FileNotFoundError: If no candidate holds ``expected_files`` recordings.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if (
            candidate.is_dir()
            and sum(
                1 for mp3 in candidate.rglob("*.mp3") if FILENAME_RE.match(mp3.name)
            )
            >= _CFG["expected_files"]
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete EmoGator audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "`data/mp3/` files, or let `prepare_source` download them from GitHub."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$EMOGATOR_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _extract_mp3s(archive: Path, destination: Path) -> None:
    """Extract the mp3 files of the tarball into ``destination``.

    Args:
        archive: ``.tar.gz`` archive to read.
        destination: Directory to extract into.

    Raises:
        tarfile.TarError: If the archive is invalid.
        ValueError: If a member would be written outside ``destination``.
    """
    logger.info("Extracting %s", archive.name)
    root = destination.resolve()
    with tarfile.open(archive) as tf:
        for member in tf:
            if not (member.isfile() and member.name.lower().endswith(".mp3")):
                continue
            if not (root / member.name).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in archive: {member.name}")
            tf.extract(member, destination, filter="data")


def _download_source(destination: Path) -> None:
    """Fetch the tarball and unpack its mp3 files.

    Args:
        destination: Directory to unpack into.

    Raises:
        urllib.error.URLError: If the download fails.
        tarfile.TarError: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / str(_CFG["archive_name"])
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    download_url(str(_CFG["archive_url"]), part, logger=logger)
    part.rename(archive_path)
    _extract_mp3s(archive_path, destination)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


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
    """Collect the recordings, taking the label from the file name.

    Args:
        source_root: Corpus root with the recordings somewhere below it.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``speaker`` and
        ``label``.

    Raises:
        ValueError: If a file's category is unknown, or two files share a
            name (the name is the utterance id).
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: dict[str, Path] = {}
    skipped = 0
    for mp3 in sorted(source_root.rglob("*.mp3")):
        fields = FILENAME_RE.match(mp3.name)
        if fields is None:
            skipped += 1
            continue
        if fields["emotion"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown category {fields['emotion']!r} in {mp3}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        if mp3.stem in seen:
            raise ValueError(f"Duplicate recording name: {mp3} and {seen[mp3.stem]}")
        seen[mp3.stem] = mp3
        entries.append(
            {
                "utt_id": f"emogator-{mp3.stem}",
                "source": str(mp3),
                "speaker": fields["speaker"],
                "label": _EMOTION_MAP[fields["emotion"]],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if not entries:
        raise FileNotFoundError(f"No EmoGator audio under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d contributor(s)",
        len(entries),
        len({e["speaker"] for e in entries}),
    )
    return _drop_duplicates(entries)


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole contributors: ``n_test_speakers`` and ``n_valid_speakers``.

    Contributors are shuffled with ``split_seed``, so every split has
    contributors the others never see.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If fewer contributors exist than the two held-out splits
            need plus one for training.
    """
    n_test = int(_CFG["n_test_speakers"])
    n_valid = int(_CFG["n_valid_speakers"])
    speakers = sorted({e["speaker"] for e in entries})
    if len(speakers) < n_test + n_valid + 1:
        raise ValueError(f"Only {len(speakers)} speakers for {n_test}+{n_valid}+train")
    Random(int(_CFG["split_seed"])).shuffle(speakers)
    split_of = {s: "test" for s in speakers[:n_test]}
    split_of.update({s: "valid" for s in speakers[n_test : n_test + n_valid]})
    return {
        e["utt_id"]: split_of.get(e["speaker"], "train")
        for e in sorted(entries, key=lambda e: e["utt_id"])
    }


def _convert(ffmpeg: str, source: str, wav: Path) -> None:
    """Convert one mp3 to a 16 kHz mono wav at ``wav`` unless it exists."""
    if wav.is_file():
        return
    part = wav.with_suffix(wav.suffix + ".part")
    subprocess.run(
        [ffmpeg, "-i", source, "-ac", "1", "-ar", str(_CFG["sampling_rate"])]
        + ["-f", "wav", "-y", "-hide_banner", "-loglevel", "error", str(part)],
        check=True,
    )
    part.replace(wav)


class EmogatorBuilder(DatasetBuilder):
    """Prepare EmoGator for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes
    one manifest per split, drawn as a contributor-held-out split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether an EmoGator source tree is reachable."""
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
        """Download and unpack the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("EmoGator already present; skipping download")
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

        entries = read_entries(source_root)
        assignment = assign_splits(entries)

        wavs = [wav_dir / f"{e['utt_id']}.wav" for e in entries]
        with ThreadPoolExecutor(int(_CFG["convert_workers"])) as pool:
            # Consume the iterator so a failed conversion raises here.
            list(
                pool.map(
                    lambda pair: _convert(ffmpeg, pair[0]["source"], pair[1]),
                    zip(entries, wavs),
                )
            )

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry, wav in zip(entries, wavs):
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
