"""AESDD dataset builder."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import urllib.request
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# `f18 (3).wav` -> emotion code f (fear), utterance 18, speaker 3. When two takes
# of one utterance were kept, a letter follows, as in `s01 (5)b.wav`.
FILENAME_RE = re.compile(
    r"^(?P<code>[a-z])(?P<item>\d{2}) \((?P<speaker>\d+)\)(?P<take>[a-z]?)\.wav$"
)

# A WAV of at most this size is a header with no samples.
_MAX_EMPTY_WAV_BYTES = 44

_MEGA_API = "https://g.api.mega.co.nz/cs"


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {
    str(k): {"label": str(v["label"]), "code": str(v["code"])}
    for k, v in _CFG["emotion_map"].items()
}


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the AESDD source tree, in order.

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
    """Return the first candidate holding at least one AESDD recording.

    Recordings are searched recursively, since the release nests the emotion
    folders under a top-level folder.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Corpus root with ``<emotion>/<code><item> (<speaker>).wav`` files
        somewhere below it.

    Raises:
        FileNotFoundError: If no candidate holds any.
    """
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if candidate.is_dir() and any(
            FILENAME_RE.match(wav.name) for wav in candidate.rglob("*.wav")
        ):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "AESDD audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding the "
        "corpus, or let `prepare_source` download it from the MEGA folder "
        "linked on the corpus page."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$AESDD_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _decode_base64(text: str) -> bytes:
    """Decode MEGA's URL-safe base64, which omits the padding."""
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _call_mega(body: list[dict], folder_handle: str) -> list:
    """Send one request to the MEGA API on behalf of a public folder link."""
    request = urllib.request.Request(
        f"{_MEGA_API}?id=1&n={folder_handle}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        reply = json.load(response)
    if isinstance(reply, int):
        raise RuntimeError(f"MEGA API error {reply}")
    return reply


def _list_mega_files(folder_handle: str, folder_key: bytes) -> list[dict]:
    """List the files of a public MEGA folder with decrypted paths and keys.

    Each node carries its own key encrypted with the folder key, and its
    attributes (the name) encrypted with that node key.

    Args:
        folder_handle: Handle from the folder link.
        folder_key: Folder key from the link, 16 bytes.

    Returns:
        list[dict]: Keys ``handle``, ``path``, ``size``, ``key`` (16 bytes) and
        ``nonce`` (8 bytes) for every file.

    Raises:
        ValueError: If a node's attributes do not decrypt.
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    def decrypt(mode, key: bytes, data: bytes) -> bytes:
        decryptor = Cipher(algorithms.AES(key), mode).decryptor()
        return decryptor.update(data) + decryptor.finalize()

    nodes = _call_mega([{"a": "f", "c": 1, "r": 1}], folder_handle)[0]["f"]
    names, parents, files = {}, {}, []
    for node in nodes:
        node_key = decrypt(
            modes.ECB(), folder_key, _decode_base64(node["k"].split(":", 1)[1])
        )
        if node["t"] == 0:
            words = struct.unpack(">8I", node_key)
            nonce = struct.pack(">2I", words[4], words[5])
            node_key = struct.pack(">4I", *(words[i] ^ words[i + 4] for i in range(4)))
        attributes = decrypt(
            modes.CBC(bytes(16)), node_key, _decode_base64(node["a"])
        ).rstrip(b"\0")
        if not attributes.startswith(b"MEGA"):
            raise ValueError(f"Cannot decrypt the name of MEGA node {node['h']}")
        names[node["h"]] = json.loads(attributes[4:])["n"]
        parents[node["h"]] = node.get("p")
        if node["t"] == 0:
            files.append(
                {
                    "handle": node["h"],
                    "size": int(node["s"]),
                    "key": node_key,
                    "nonce": nonce,
                }
            )
    for entry in files:
        parts, handle = [], entry["handle"]
        while handle in names:
            parts.append(names[handle])
            handle = parents[handle]
        entry["path"] = "/".join(reversed(parts))
    return files


def _download_mega_file(entry: dict, folder_handle: str, destination: Path) -> None:
    """Download one MEGA file and decrypt it (AES-CTR) to ``destination``."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    url = _call_mega([{"a": "g", "g": 1, "n": entry["handle"]}], folder_handle)[0]["g"]
    decryptor = Cipher(
        algorithms.AES(entry["key"]), modes.CTR(entry["nonce"] + bytes(8))
    ).decryptor()
    part = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(decryptor.update(chunk))
        fh.write(decryptor.finalize())
    if part.stat().st_size != entry["size"]:
        raise RuntimeError(
            f"{entry['path']}: got {part.stat().st_size} bytes, expected "
            f"{entry['size']}"
        )
    part.replace(destination)


def _download_source(destination: Path) -> None:
    """Fetch every file of the corpus's public MEGA folder.

    Files that already have the expected size are kept, so an interrupted
    download resumes where it stopped.

    Args:
        destination: Directory to download into.

    Raises:
        RuntimeError: If MEGA refuses a request or a file arrives truncated.
        urllib.error.URLError: If a download fails.
    """
    handle = str(_CFG["mega_folder_handle"])
    files = _list_mega_files(handle, _decode_base64(str(_CFG["mega_folder_key"])))
    logger.info("Downloading %d file(s) from MEGA folder %s", len(files), handle)
    for number, entry in enumerate(files, start=1):
        target = destination / entry["path"]
        if not (target.is_file() and target.stat().st_size == entry["size"]):
            _download_mega_file(entry, handle, target)
        if number % 50 == 0:
            logger.info("Downloaded %d/%d files", number, len(files))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the emotion folder.

    Args:
        source_root: Corpus root with the emotion folders somewhere below it.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source`` and ``label``.

    Raises:
        ValueError: If a recording sits in an unknown folder, its file name
            code disagrees with the folder, or two recordings get the same id.
        FileNotFoundError: If no recording is found.
    """
    entries = []
    seen: dict[str, Path] = {}
    skipped = 0
    for wav in sorted(source_root.rglob("*.wav")):
        if wav.stat().st_size <= _MAX_EMPTY_WAV_BYTES:
            logger.warning("Skipping %s: it holds no audio samples", wav)
            continue
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            skipped += 1
            continue
        emotion = wav.parent.name
        if emotion not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion folder {emotion!r} for {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        if fields["code"] != _EMOTION_MAP[emotion]["code"]:
            raise ValueError(f"{wav} does not start with the code of {emotion!r}")
        utt_id = (
            f"aesdd-{emotion}-{fields['code']}{fields['item']}"
            f"_{fields['speaker']}{fields['take']}"
        )
        if utt_id in seen:
            raise ValueError(f"Duplicate recording id: {wav} and {seen[utt_id]}")
        seen[utt_id] = wav
        entries.append(
            {
                "utt_id": utt_id,
                "source": str(wav),
                "label": _EMOTION_MAP[emotion]["label"],
            }
        )
    if skipped:
        logger.warning("Skipped %d file(s) with an unexpected name", skipped)
    if not entries:
        raise FileNotFoundError(f"No AESDD audio under {source_root}")
    logger.info("Read %d utterance(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. Speakers are not held out.

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


class AESDDBuilder(DatasetBuilder):
    """Prepare AESDD for the classification recipe.

    `build()` converts every recording to 16 kHz mono and writes one manifest
    per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a AESDD source tree is reachable."""
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
            logger.info("AESDD already present; skipping download")
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
