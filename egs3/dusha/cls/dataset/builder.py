"""Dusha dataset builder."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import ssl
import subprocess
import tarfile
import urllib.error
import urllib.request
import wave
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
_SETUP_FILES = {str(k): Path(str(v)) for k, v in _CFG["setup_files"].items()}
_MARKUP_FILES = {str(k): Path(str(v)) for k, v in _CFG["markup_files"].items()}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the Dusha source tree, in order.

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


def _wav_path(root: Path, split: str, hash_id: str) -> Path:
    return root / str(_CFG["wav_dirs"][split]) / f"{hash_id}.wav"


def _read_setup(root: Path, split: str) -> list[tuple[str, str]]:
    """Read the ``(hash_id, emotion)`` rows of one official setup file."""
    rows = []
    with (root / _SETUP_FILES[split]).open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            rows.append((str(row["id"]), str(row["emotion"])))
    return rows


def _complete_archive(candidate: Path) -> Path | None:
    """Return ``candidate/crowd.tar`` if it has its full size, else None."""
    archive = candidate / "crowd.tar"
    if archive.is_file() and archive.stat().st_size == int(_CFG["crowd_bytes"]):
        return archive
    return None


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding the label files and all the audio.

    The audio is either the complete ``crowd.tar``, which is read in place, or
    an extracted WAV for every utterance the setup files list.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with both setup files, both markup files and the
        audio.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        files = [*_SETUP_FILES.values(), *_MARKUP_FILES.values()]
        if candidate.is_dir() and all((candidate / f).is_file() for f in files):
            if _complete_archive(candidate) is not None or all(
                _wav_path(candidate, split, hash_id).is_file()
                for split in _SETUP_FILES
                for hash_id, _ in _read_setup(candidate, split)
            ):
                return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete Dusha data not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding `crowd.tar` "
        "and the label files (or the extracted WAVs), or let `prepare_source` "
        "download them."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$DUSHA_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


_ALLOW_EXPIRED_ENV = "DUSHA_ALLOW_EXPIRED_CERTIFICATE"
# OpenSSL's X509_V_FLAG_NO_CHECK_TIME (not exported by the ssl module).
_X509_V_FLAG_NO_CHECK_TIME = 0x200000


def _allow_expired_certificate() -> bool:
    env_value = os.environ.get(_ALLOW_EXPIRED_ENV, "").strip().lower()
    if env_value:
        return env_value in ("1", "true", "yes")
    return bool(_CFG.get("allow_expired_certificate", False))


def _ssl_context() -> ssl.SSLContext:
    """Return the TLS context for the CDN.

    The certificate chain and the host name are always verified. With
    ``allow_expired_certificate`` (or ``$DUSHA_ALLOW_EXPIRED_CERTIFICATE=1``)
    only the validity dates are not checked, so a certificate that has expired
    is accepted while a certificate for another host or from an untrusted
    issuer is still refused.
    """
    context = ssl.create_default_context()
    if _allow_expired_certificate():
        context.verify_flags |= _X509_V_FLAG_NO_CHECK_TIME
    return context


def _download_checked(url: str, destination: Path, size: int) -> Path:
    """Download ``url`` to ``destination`` and check its size.

    A partial download (``<name>.part``) is resumed. No checksum is published
    (the CDN's ETag is a multipart hash), so the byte count is the only
    completeness check.

    Raises:
        ssl.SSLCertVerificationError: If the certificate is refused; an
            expired one names the setting that accepts it.
        ValueError: If the downloaded file has the wrong size.
    """
    part = destination.with_suffix(destination.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        if _allow_expired_certificate():
            logger.warning(
                "Not checking the validity dates of the server certificate "
                "(chain and host name are still verified): %s",
                url,
            )
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={have}-"} if have else {}
        )
        logger.info("Downloading %s from byte %d of %d", destination.name, have, size)
        try:
            response = urllib.request.urlopen(request, timeout=120, context=_ssl_context())
        except urllib.error.URLError as err:
            reason = getattr(err, "reason", None)
            if isinstance(reason, ssl.SSLCertVerificationError) and (
                "expired" in str(reason)
            ):
                raise ssl.SSLCertVerificationError(
                    f"The certificate of {url} has expired. To accept it while "
                    "still checking its issuer and host name, set "
                    f"`allow_expired_certificate: true` in dataset/config.yaml "
                    f"or ${_ALLOW_EXPIRED_ENV}=1 (see readme.md)."
                ) from err
            raise
        with response, part.open("ab" if have else "wb") as fh:
            if have and response.status != 206:
                raise ValueError(f"{url}: the server ignored the resume request")
            step, next_report = size // 20, have + size // 20
            while chunk := response.read(1 << 20):
                fh.write(chunk)
                have += len(chunk)
                if have >= next_report:
                    logger.info("%s: %d%%", destination.name, have * 100 // size)
                    next_report += step
    if part.stat().st_size != size:
        raise ValueError(
            f"{destination.name}: got {part.stat().st_size} bytes, expected {size}"
        )
    part.rename(destination)
    return destination


def _extract_setups(archive: Path, destination: Path) -> None:
    """Write the two setup files of ``paper_setups`` into ``destination``.

    The archive is a plain tar although it is named ``.tgz``. Members are read
    by name and written to paths built here, so no archive path is trusted.
    """
    wanted = {str(path): path for path in _SETUP_FILES.values()}
    with tarfile.open(archive, "r:*") as tar:
        for name, relpath in wanted.items():
            member = tar.extractfile(name)
            if member is None:
                raise FileNotFoundError(f"{name} is missing from {archive.name}")
            target = destination / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            with member, target.open("wb") as dst:
                shutil.copyfileobj(member, dst)


def _extract_markup(archive: Path, destination: Path) -> None:
    """Write the two markup files of ``crowd.tar`` into ``destination``.

    The recordings stay in the archive and are read from it by `build()`.
    Target paths are built here, not taken from the archive.
    """
    wanted = {str(path): destination / path for path in _MARKUP_FILES.values()}
    seen = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            target = wanted.get(member.name)
            if target is None or not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            seen += 1
    if seen != len(wanted):
        raise ValueError(f"{archive.name}: found {seen} of {len(wanted)} markup files")


def _index_archive(archive: Path) -> dict[str, tuple[int, int]]:
    """Map each WAV member name of ``crowd.tar`` to its ``(offset, size)``.

    The archive is not compressed, so a member's bytes can be read later at
    its offset without extracting anything.
    """
    index = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".wav"):
                index[member.name] = (member.offset_data, member.size)
    return index


def _download_source(destination: Path) -> None:
    """Fetch the setup files and the crowd archive.

    ``crowd.tar`` is kept as it is; only its two markup files are written next
    to it.

    Args:
        destination: Directory to download into.

    Raises:
        urllib.error.URLError: If a download fails.
        ValueError: If a download has the wrong size or the archive lacks files.
    """
    destination.mkdir(parents=True, exist_ok=True)
    if not all((destination / f).is_file() for f in _SETUP_FILES.values()):
        setups = _download_checked(
            str(_CFG["setups_url"]),
            destination / "paper_setups.tgz",
            int(_CFG["setups_bytes"]),
        )
        _extract_setups(setups, destination)
        setups.unlink()
    crowd = _complete_archive(destination) or _download_checked(
        str(_CFG["crowd_url"]),
        destination / "crowd.tar",
        int(_CFG["crowd_bytes"]),
    )
    if not all((destination / f).is_file() for f in _MARKUP_FILES.values()):
        _extract_markup(crowd, destination)


def _read_source_ids(path: Path) -> dict[str, str]:
    """Map each ``hash_id`` of a markup file to its ``source_id``.

    The markup has one row per annotation, so a ``hash_id`` repeats. The first
    column is ``hash_id`` and the last is ``source_id`` (empty for the
    control tasks); the free-text columns in between may contain anything, so
    only the two ends of each line are read.
    """
    source_ids: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        if header[0] != "hash_id" or header[-1] != "source_id":
            raise ValueError(f"Unexpected header in {path}: {header}")
        for line in fh:
            fields = line.rstrip("\r\n").split("\t")
            source_ids.setdefault(
                fields[0], "" if fields[-1] in ("", "nan") else fields[-1]
            )
    return source_ids


def _drop_duplicates(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove the listed same-sound, same-label utterances if switched on.

    Applies only when ``exclude_duplicates`` is true. Each listed utterance has
    the same sound and label as one that is kept.

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
    """Collect the utterances of the two official setup files.

    Args:
        source_root: Corpus root holding the setup files, markup and audio
            (``crowd.tar`` or the extracted WAVs).

    Returns:
        list[dict]: Keys ``utt_id``, ``source_id``, ``split`` (``train`` or
        ``test``, as published), ``label`` and the audio: ``archive``,
        ``offset`` and ``size`` when read from ``crowd.tar``, else ``source``
        (a WAV path).

    Raises:
        ValueError: If an emotion is not in ``emotion_map`` or an utterance
            appears twice.
        FileNotFoundError: If a WAV is missing.
    """
    archive = _complete_archive(source_root)
    index = _index_archive(archive) if archive is not None else None
    entries = []
    seen: set[str] = set()
    for split in _SETUP_FILES:
        source_ids = _read_source_ids(source_root / _MARKUP_FILES[split])
        for hash_id, emotion in _read_setup(source_root, split):
            if emotion not in _EMOTION_MAP:
                raise ValueError(f"Unknown emotion {emotion!r} for {hash_id}")
            if hash_id in seen:
                raise ValueError(f"Duplicate utterance: {hash_id}")
            seen.add(hash_id)
            entry = {
                "utt_id": f"dusha-{hash_id}",
                "source_id": source_ids[hash_id],
                "split": split,
                "label": _EMOTION_MAP[emotion],
            }
            if index is not None:
                name = f"{_CFG['wav_dirs'][split]}/{hash_id}.wav"
                if name not in index:
                    raise FileNotFoundError(f"{name} is missing from {archive}")
                entry.update(
                    archive=str(archive), offset=index[name][0], size=index[name][1]
                )
            else:
                wav = _wav_path(source_root, split, hash_id)
                if not wav.is_file():
                    raise FileNotFoundError(wav)
                entry["source"] = str(wav)
            entries.append(entry)
    conflicting = {str(i) for i in _CFG["conflicting_ids"]}
    missing = conflicting - {e["utt_id"] for e in entries}
    if missing:
        raise ValueError(f"`conflicting_ids` not found: {sorted(missing)}")
    entries = [e for e in entries if e["utt_id"] not in conflicting]
    entries = _drop_duplicates(entries)
    logger.info("Read %d utterance(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Keep the official test set and hold out part of train for validation.

    Whole ``source_id`` groups are moved to ``valid`` (in the order of a
    shuffle with ``split_seed``) until ``valid_ratio`` of the train utterances
    is reached, so the validation set shares no ``source_id`` with train. An
    utterance without a ``source_id`` is its own group.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If ``valid_ratio`` is not in (0, 1).
    """
    valid_ratio = float(_CFG["valid_ratio"])
    if not 0 < valid_ratio < 1:
        raise ValueError(f"`valid_ratio` must be in (0, 1), got {valid_ratio}")

    groups: dict[str, list[str]] = {}
    assignment: dict[str, str] = {}
    for entry in sorted(entries, key=lambda e: e["utt_id"]):
        if entry["split"] == "test":
            assignment[entry["utt_id"]] = "test"
        else:
            key = entry["source_id"] or entry["utt_id"]
            groups.setdefault(key, []).append(entry["utt_id"])

    target = round(valid_ratio * sum(len(ids) for ids in groups.values()))
    order = sorted(groups)
    Random(int(_CFG["split_seed"])).shuffle(order)
    n_valid = 0
    for key in order:
        split = "valid" if n_valid < target else "train"
        if split == "valid":
            n_valid += len(groups[key])
        for utt_id in groups[key]:
            assignment[utt_id] = split
    return assignment


def _is_target_format(path) -> bool:
    """Tell whether ``path`` (a path or a file object) is 16 kHz mono 16-bit WAV."""
    try:
        with wave.open(path if hasattr(path, "read") else str(path), "rb") as wav:
            return (
                wav.getframerate() == int(_CFG["sampling_rate"])
                and wav.getnchannels() == 1
                and wav.getsampwidth() == 2
            )
    except (wave.Error, EOFError):
        return False


class DushaBuilder(DatasetBuilder):
    """Prepare Dusha for the classification recipe.

    `prepare_source` downloads ``crowd.tar`` and keeps it as it is (only the
    label files are written next to it). `build()` keeps the published
    train/test partition of the crowd domain and carves the validation set out
    of train (see :func:`assign_splits`). It reads each listed recording from
    ``crowd.tar`` and writes it into the output directory (converted with
    ffmpeg if it is not 16 kHz mono 16-bit). With an extracted tree instead of
    the archive, a recording in the target format is referenced where it lies.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete Dusha source tree is reachable."""
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
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("Dusha already present; skipping download")
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
        """Write one TSV manifest per split, converting audio only if needed.

        The format of the recordings is checked by ``num_workers`` threads at a
        time.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus is incomplete, or ``ffmpeg`` is
                needed and cannot be found.
            subprocess.CalledProcessError: If a conversion fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)
        wav_dir = data_root / str(_CFG["wav_subdir"])
        sampling_rate = str(_CFG["sampling_rate"])

        entries = read_entries(source_root)
        assignment = assign_splits(entries)

        def locate(entry: dict) -> Path:
            wav = wav_dir / f"{entry['utt_id']}.wav"
            data = None
            if "archive" in entry:
                # Read from crowd.tar: the recording is written to the output
                # directory, as it is if it already has the target format.
                if wav.is_file():
                    return wav
                with open(entry["archive"], "rb") as fh:
                    fh.seek(entry["offset"])
                    data = fh.read(entry["size"])
                if len(data) != entry["size"]:
                    raise ValueError(f"Truncated archive: {entry['archive']}")
                if _is_target_format(io.BytesIO(data)):
                    wav_dir.mkdir(parents=True, exist_ok=True)
                    part = wav.with_suffix(wav.suffix + ".part")
                    part.write_bytes(data)
                    part.replace(wav)
                    return wav
                source_arg = "pipe:0"
            else:
                source = Path(entry["source"])
                if _is_target_format(source):
                    return source
                source_arg = str(source)
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise FileNotFoundError(
                    f"ffmpeg is required to convert {entry['utt_id']} to 16 kHz mono"
                )
            if not wav.is_file():
                wav_dir.mkdir(parents=True, exist_ok=True)
                part = wav.with_suffix(wav.suffix + ".part")
                subprocess.run(
                    [
                        ffmpeg,
                        "-i",
                        source_arg,
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
                    input=data,
                    check=True,
                )
                part.replace(wav)
            return wav

        with ThreadPoolExecutor(max_workers=int(_CFG["num_workers"])) as pool:
            wavs = list(pool.map(locate, entries))
        n_written = sum(1 for wav in wavs if wav.is_relative_to(wav_dir))
        logger.info(
            "%d of %d recordings written to %s (the others are referenced in place)",
            n_written,
            len(wavs),
            wav_dir,
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
