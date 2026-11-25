"""SantaliESC dataset builder."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator
from urllib.parse import quote

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)

# How many paths an error message or log line shows.
_SAMPLE_SIZE = 20


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_AUDIO_EXTENSIONS = {str(e).lower() for e in _CFG["audio_extensions"]}
_IGNORE_RES = [re.compile(str(p)) for p in _CFG.get("ignore_path_re") or []]
# Alias (lowercase) -> label.
_ALIASES: dict[str, str] = {
    str(alias).lower(): str(label)
    for label, aliases in _CFG["emotion_aliases"].items()
    for alias in aliases
}
_LABELS = sorted({str(label) for label in _CFG["emotion_aliases"]})
_EMOTION_RE = re.compile(str(_CFG["emotion_re"])) if _CFG.get("emotion_re") else None
_EMOTION_CODES = {str(k): str(v) for k, v in (_CFG.get("emotion_codes") or {}).items()}
_SPEAKER_RE = re.compile(str(_CFG["speaker_re"])) if _CFG.get("speaker_re") else None


def _words(text: str) -> list[str]:
    """Lowercase ``text`` and cut it into runs of letters and runs of digits."""
    return re.findall(r"[a-z]+|[0-9]+", text.lower())


def _sample(items: list[str]) -> str:
    shown = items[:_SAMPLE_SIZE]
    more = f"\n  ... ({len(items) - len(shown)} more)" if len(items) > len(shown) else ""
    return "\n  " + "\n  ".join(shown) + more


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the SantaliESC source tree, in order.

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


def list_source_files(source_root: Path) -> list[str]:
    """List every file under ``source_root`` as a sorted relative POSIX path."""
    return sorted(
        p.relative_to(source_root).as_posix()
        for p in source_root.rglob("*")
        if p.is_file()
    )


def _is_audio(relpath: str) -> bool:
    if any(r.search(relpath) for r in _IGNORE_RES):
        return False
    return Path(relpath).suffix.lower() in _AUDIO_EXTENSIONS


def _has_audio(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    return any(
        p.is_file() and _is_audio(p.relative_to(candidate).as_posix())
        for p in candidate.rglob("*")
    )


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding at least one audio file (any depth).

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root.

    Raises:
        FileNotFoundError: If no candidate holds any audio file.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if _has_audio(candidate):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "SantaliESC audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "unpacked archive, or let `prepare_source` download it from Zenodo."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$SANTALIESC_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    part.replace(path)


def _verify_md5(archive: Path, expected: str) -> None:
    """Check a downloaded archive against the checksum Zenodo publishes.

    Raises:
        ValueError: If the digest does not match.
    """
    digest = hashlib.md5()
    with archive.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"{archive.name} has md5 {actual}, but the Zenodo record lists "
            f"{expected}. Delete it and let this stage fetch it again."
        )
    logger.info("Checksum verified: %s", actual)


def _extract_zip(archive: Path, destination: Path, members_file: Path) -> None:
    """Extract one zip archive into ``destination`` and list its members.

    ``espnet3.utils.download_utils`` only extracts gzipped tar archives, so
    this is done here. Members are checked against the destination first:
    ``ZipFile.extractall`` does not reject absolute paths or ``..``
    components on its own. The member list (name and uncompressed size) is
    written to ``members_file`` before extraction, so the layout can be
    inspected even when a later step fails.

    Raises:
        ValueError: If a member would be written outside ``destination``.
        zipfile.BadZipFile: If the archive is invalid.
    """
    logger.info("Extracting %s", archive.name)
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        _write_lines(members_file, [f"{i.filename}\t{i.file_size}" for i in infos])
        logger.info("Listed %d archive member(s) in %s", len(infos), members_file)
        for info in infos:
            if not (destination / info.filename).resolve().is_relative_to(destination):
                raise ValueError(
                    f"{archive.name} escapes its destination: {info.filename}"
                )
        zf.extractall(path=destination)


def _download_source(destination: Path, members_file: Path) -> None:
    """Fetch, verify and unpack the corpus archive.

    Raises:
        urllib.error.URLError: If the download fails.
        ValueError: If the checksum does not match.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_name = str(_CFG["archive_name"])
    # Zenodo's file-content endpoint, not the plain `/files/<name>` path
    # (which returns file metadata as JSON). The name holds spaces.
    url = (
        str(_CFG["base_url"]).rstrip("/")
        + "/"
        + quote(archive_name, safe="()")
        + "/content"
    )

    archive_path = destination / archive_name
    part = archive_path.with_suffix(archive_path.suffix + ".part")
    logger.info("Downloading %s (about 890 MB)", archive_name)
    download_url(url, part, logger=logger)
    part.rename(archive_path)
    _verify_md5(archive_path, str(_CFG["archive_md5"]))
    _extract_zip(archive_path, destination, members_file)
    if _CFG.get("remove_archive", False):
        archive_path.unlink(missing_ok=True)


def parse_emotion(relpath: str) -> str:
    """Return the label of one audio file from its relative path.

    Args:
        relpath: Path relative to the source root, ``/``-separated.

    Returns:
        str: One of the keys of ``emotion_aliases``.

    Raises:
        ValueError: If no emotion, or more than one, is found.
    """
    stem = relpath[: -len(Path(relpath).suffix)] if Path(relpath).suffix else relpath
    if _EMOTION_RE is not None:
        match = _EMOTION_RE.search(stem)
        if match is None:
            raise ValueError(f"`emotion_re` does not match: {relpath}")
        value = match["emotion"]
        if value in _EMOTION_CODES:
            return _EMOTION_CODES[value]
        if value.lower() in _ALIASES:
            return _ALIASES[value.lower()]
        raise ValueError(f"Unknown emotion {value!r} in {relpath}")
    found = {_ALIASES[w] for w in _words(stem) if w in _ALIASES}
    if len(found) != 1:
        what = "no emotion" if not found else f"several emotions {sorted(found)}"
        raise ValueError(f"{what} in {relpath}")
    return found.pop()


def _folder_speaker(relpath: str) -> str | None:
    """Return the innermost folder of ``relpath`` that is not an emotion name."""
    for folder in reversed(relpath.split("/")[:-1]):
        if not any(w in _ALIASES for w in _words(folder)):
            return folder
    return None


def assign_speakers(entries: list[dict[str, str]]) -> bool:
    """Fill ``entry["speaker"]`` when the layout identifies speakers.

    Uses ``speaker_re`` when set (it must match every file); otherwise the
    folder rule described in the config, accepted only when it yields exactly
    ``expected_speakers`` folders that each hold every label.

    Returns:
        bool: Whether speakers were assigned.

    Raises:
        ValueError: If ``speaker_re`` is set and misses a file.
    """
    if _SPEAKER_RE is not None:
        missing = []
        for entry in entries:
            match = _SPEAKER_RE.search(entry["relpath"])
            if match is None:
                missing.append(entry["relpath"])
            else:
                entry["speaker"] = match["speaker"]
        if missing:
            raise ValueError(
                f"`speaker_re` does not match {len(missing)} file(s):"
                + _sample(missing)
            )
        return True

    if not _CFG.get("infer_speaker_from_folders", False):
        return False
    folders = [_folder_speaker(e["relpath"]) for e in entries]
    labels_per_folder: dict[str | None, set[str]] = {}
    for folder, entry in zip(folders, entries):
        labels_per_folder.setdefault(folder, set()).add(entry["label"])
    expected = int(_CFG["expected_speakers"])
    if (
        None not in labels_per_folder
        and len(labels_per_folder) == expected
        and all(labels == set(_LABELS) for labels in labels_per_folder.values())
    ):
        for folder, entry in zip(folders, entries):
            entry["speaker"] = str(folder)
        logger.info("Speakers taken from folder names (%d speakers)", expected)
        return True
    logger.warning(
        "Speakers not identified from folders: found %d candidate folder(s), "
        "expected %d each holding all of %s. Sample: %s",
        len(labels_per_folder),
        expected,
        _LABELS,
        sorted(str(f) for f in labels_per_folder)[:_SAMPLE_SIZE],
    )
    return False


def read_entries(source_root: Path, all_files: list[str]) -> list[dict[str, str]]:
    """Collect the audio files and label them from their paths.

    Args:
        source_root: Corpus root.
        all_files: Every file under ``source_root`` (relative POSIX paths).

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``relpath``,
        ``label`` and, when known, ``speaker``.

    Raises:
        FileNotFoundError: If no audio file is found.
        ValueError: If any audio file cannot be labelled; the message shows a
            sample of the paths so the rules in ``config.yaml`` can be fixed.
    """
    audio = [p for p in all_files if _is_audio(p)]
    audio_set = set(audio)
    other = Counter(
        Path(p).suffix.lower() or "<none>" for p in all_files if p not in audio_set
    )
    if other:
        logger.info("Not read (by extension): %s", dict(other))
    if not audio:
        raise FileNotFoundError(
            f"No audio ({sorted(_AUDIO_EXTENSIONS)}) under {source_root}. Files "
            "found:" + _sample(all_files)
        )

    entries, failures = [], []
    for relpath in audio:
        try:
            label = parse_emotion(relpath)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        utt_id = "santaliesc-" + re.sub(r"[^0-9A-Za-z-]+", "_", Path(relpath).stem)
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(source_root / relpath),
                "relpath": relpath,
                "label": label,
            }
        )
    if failures:
        raise ValueError(
            f"{len(failures)} of {len(audio)} audio file(s) could not be "
            "labelled; adjust `emotion_aliases`/`emotion_re` in "
            "dataset/config.yaml:" + _sample(failures)
            + "\nSample of the audio paths found:" + _sample(audio)
        )
    duplicated = [u for u, n in Counter(e["utt_id"] for e in entries).items() if n > 1]
    if duplicated:
        raise ValueError(f"Utterance ids collide:{_sample(sorted(duplicated))}")

    entries = _drop_listed(entries)

    expected = int(_CFG["expected_utterances"])
    if len(entries) != expected and not (
        _CFG["exclude_duplicates"] or _CFG["exclude_conflicting"]
    ):
        logger.warning(
            "Found %d utterances, but the Zenodo record states %d",
            len(entries),
            expected,
        )
    logger.info(
        "Read %d utterance(s); per label: %s",
        len(entries),
        dict(sorted(Counter(e["label"] for e in entries).items())),
    )
    return entries


def _drop_listed(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Leave out the recordings listed in ``duplicate_ids``/``conflicting_ids``.

    Each list applies only when its ``exclude_*`` switch is true. Every listed
    id is checked against the release, so a stale list stops the build.

    Raises:
        ValueError: If a listed id is not among ``entries``.
    """
    ids = {e["utt_id"] for e in entries}
    listed = {str(i) for i in _CFG["duplicate_ids"]} | {
        str(i) for i in _CFG["conflicting_ids"]
    }
    missing = listed - ids
    if missing:
        raise ValueError(f"Listed duplicate ids not found: {sorted(missing)}")
    excluded: set[str] = set()
    if _CFG["exclude_duplicates"]:
        excluded |= {str(i) for i in _CFG["duplicate_ids"]}
    if _CFG["exclude_conflicting"]:
        excluded |= {str(i) for i in _CFG["conflicting_ids"]}
    if excluded:
        logger.info("Left out %d listed duplicate recording(s)", len(excluded))
    return [e for e in entries if e["utt_id"] not in excluded]


def _check_ratios(prefix: str) -> tuple[float, float, float]:
    ratios = tuple(
        float(_CFG[f"{prefix}{name}_ratio"]) for name in ("train", "valid", "test")
    )
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(
            f"`{prefix}train_ratio`/`{prefix}valid_ratio`/`{prefix}test_ratio` "
            f"must sum to 1, got {ratios}"
        )
    return ratios


def assign_splits_by_utterance(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label (speakers shared)."""
    _, valid_ratio, test_ratio = _check_ratios("")
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


def assign_splits_by_speaker(entries: list[dict[str, str]]) -> dict[str, str]:
    """Shuffle the speakers with ``split_seed`` and cut them by the ratios."""
    _, valid_ratio, test_ratio = _check_ratios("speaker_")
    speakers = sorted({e["speaker"] for e in entries})
    Random(int(_CFG["split_seed"])).shuffle(speakers)
    n_valid = round(len(speakers) * valid_ratio)
    n_test = round(len(speakers) * test_ratio)
    split_of = {s: "test" for s in speakers[:n_test]}
    split_of.update({s: "valid" for s in speakers[n_test : n_test + n_valid]})
    split_of.update({s: "train" for s in speakers[n_test + n_valid :]})
    logger.info(
        "Speaker-independent split: %s speakers",
        dict(sorted(Counter(split_of.values()).items())),
    )
    return {e["utt_id"]: split_of[e["speaker"]] for e in entries}


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Pick the split mode from ``split_mode`` and the speakers found.

    Raises:
        ValueError: If ``split_mode`` is ``speaker`` but no speaker is known,
            or the mode is unknown.
    """
    mode = str(_CFG["split_mode"])
    if mode not in ("auto", "speaker", "utterance"):
        raise ValueError(f"Unknown `split_mode`: {mode}")
    has_speakers = mode != "utterance" and assign_speakers(entries)
    if mode == "speaker" and not has_speakers:
        raise ValueError(
            "`split_mode: speaker` but no speaker could be identified; set "
            "`speaker_re` in dataset/config.yaml"
        )
    if has_speakers:
        return assign_splits_by_speaker(entries)
    logger.warning("No speakers known: random label-stratified utterance split")
    return assign_splits_by_utterance(entries)


class SantaliESCBuilder(DatasetBuilder):
    """Prepare SantaliESC for the classification recipe.

    The archive layout is not documented, so `build()` lists every file it
    finds, labels the audio from its path (rules in ``config.yaml``),
    converts it to 16 kHz mono and writes one manifest per split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a SantaliESC source tree is reachable."""
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
            logger.info("SantaliESC already present; skipping download")
            return

        destination = next(iter_source_candidates(recipe_root, source_dir))
        members_file = resolve_data_root(recipe_root) / str(
            _CFG["archive_members_file"]
        )
        _download_source(destination, members_file)
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
            ValueError: If the layout does not match the labelling rules.
            subprocess.CalledProcessError: If a resample fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        # Written before parsing so the layout can be inspected on failure.
        all_files = list_source_files(source_root)
        listing = data_root / str(_CFG["source_files_file"])
        _write_lines(listing, all_files)
        logger.info("Listed %d source file(s) in %s", len(all_files), listing)

        entries = read_entries(source_root, all_files)
        assignment = assign_splits(entries)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError(
                "ffmpeg is required to resample the recordings to 16 kHz"
            )

        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)
        sampling_rate = str(_CFG["sampling_rate"])

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
                        "-c:a",
                        "pcm_s16le",
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
            _write_lines(
                manifest,
                [f"{u}\t{w}\t{lab}" for u, w, lab in sorted(rows[split])],
            )
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
