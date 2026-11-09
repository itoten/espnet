"""C-BESD dataset builder (6 classes, language-pooled, grouped split).

Takes the label from the folder, drops files whose name disagrees or whose
content is shared between labels, and resamples the rest to 16 kHz mono into
`$CBESD_OUTPUT/wav`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import zipfile
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

# `1.TF_12_angry_1.wav`, `1.EF_12 Angry_1.wav`, `3EF_9 Angry_2.wav`: child
# number, language, sex, age (per the Kaggle description), emotion, take. Separators
# vary, and one file ends in ` - Copy`.
NAME_RE = re.compile(
    r"^(?P<num>\d+)\.?\s*(?P<lang>[TE])(?P<sex>[FM])[_-]\s*(?P<age>\d+)\s*[_ ]?\s*"
    r"(?P<emotion>[A-Za-z]+)\s*[_ -]?\s*(?P<take>\d+)(?: - Copy)?\.wav$"
)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}
_LANG_CODES = {str(k): str(v) for k, v in _CFG["language_codes"].items()}


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
    return sum(1 for _ in audio_dir.glob("*/*/*.wav")) if audio_dir.is_dir() else 0


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
        "C-BESD audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + "\n\nLet `prepare_source` download it with the Kaggle CLI (an API token "
        "in ~/.kaggle/access_token or $KAGGLE_API_TOKEN), or download\n  "
        f"{_CFG['request_url']}\n"
        f"and set ${_CFG['source_env_var']} to a directory holding "
        f"{_CFG['audio_subdir']}/ with {_CFG['expected_files']} wav files "
        "below it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory holding the manifests, token list and WAV files.

    Returns:
        Path: ``$CBESD_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / str(_CFG["data_path"])


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the folder.

    Files whose name spells another emotion than the folder, byte-identical
    files filed under different labels, and `- Copy` duplicates are dropped.
    Of byte-identical files with one label, only the first utterance id is
    kept (each count is logged).

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``group``, ``lang``,
        ``digest`` and ``label``.

    Raises:
        ValueError: If a name or folder is unexpected, or two files map to the
            same utterance id with different content.
        FileNotFoundError: If no recording is found.
    """
    entries: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    n_mismatch = n_copy = 0
    audio_dir = source_root / str(_CFG["audio_subdir"])
    for wav in sorted(audio_dir.glob("*/*/*.wav")):
        lang_dir, emo_dir = wav.parent.parent.name, wav.parent.name
        if lang_dir not in _LANG_CODES or emo_dir not in _EMOTION_MAP:
            raise ValueError(f"Unexpected folder for {wav}")
        fields = NAME_RE.match(wav.name)
        if fields is None:
            raise ValueError(f"Unexpected recording name {wav.name!r}")
        folder_prefix = _EMOTION_MAP[emo_dir][:2]
        if fields["emotion"].lower()[:2] != folder_prefix:
            n_mismatch += 1
            continue
        lang = _LANG_CODES[lang_dir]
        utt_id = (
            f"cbesd-{lang}-{emo_dir.lower()}-{fields['num']}-"
            f"{fields['sex']}{fields['age']}-{fields['take']}"
        )
        entry = {
            "utt_id": utt_id,
            "source": str(wav),
            "group": f"{lang}-{fields['sex']}{fields['age']}-{fields['num']}",
            "lang": lang,
            "digest": _md5(wav),
            "label": _EMOTION_MAP[emo_dir],
        }
        if utt_id in by_id:
            if by_id[utt_id]["digest"] != entry["digest"]:
                raise ValueError(f"{wav} and {by_id[utt_id]['source']} share an id")
            n_copy += 1
            continue
        by_id[utt_id] = entry
        entries.append(entry)

    labels_of: dict[str, set[str]] = {}
    for e in entries:
        labels_of.setdefault(e["digest"], set()).add(e["label"])
    n_total = len(entries)
    entries = [e for e in entries if len(labels_of[e["digest"]]) == 1]
    n_conflict = n_total - len(entries)
    # Same audio under one label but different ids: keep the first id.
    first_of: dict[str, str] = {}
    for e in sorted(entries, key=lambda e: e["utt_id"]):
        first_of.setdefault(e["digest"], e["utt_id"])
    n_before = len(entries)
    entries = [e for e in entries if first_of[e["digest"]] == e["utt_id"]]
    n_same = n_before - len(entries)
    if n_mismatch or n_copy or n_conflict or n_same:
        logger.warning(
            "Dropped %d file(s) whose name disagrees with the folder, %d duplicate "
            "copy file(s), %d file(s) whose audio appears under two labels and "
            "%d repeated file(s) of the same audio and label",
            n_mismatch,
            n_copy,
            n_conflict,
            n_same,
        )
    if not entries:
        raise FileNotFoundError(f"No C-BESD audio under {source_root}")
    logger.info("Read %d utterance(s)", len(entries))
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Assign whole groups to splits so every language/label keeps the ratios.

    Recordings are grouped by ``group``; groups that share byte-identical audio
    are merged. Groups are taken largest first (ties broken with ``split_seed``),
    each going to the split that is furthest below its per-language, per-label targets.

    Returns:
        dict[str, str]: Utterance id to split name.
    """
    ratios = {
        "train": float(_CFG["train_ratio"]),
        "valid": float(_CFG["valid_ratio"]),
        "test": float(_CFG["test_ratio"]),
    }
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_group_of_digest: dict[str, str] = {}
    for e in entries:
        first = first_group_of_digest.setdefault(e["digest"], e["group"])
        parent[find(e["group"])] = find(first)

    members: dict[str, list[dict[str, str]]] = {}
    for e in entries:
        members.setdefault(find(e["group"]), []).append(e)
    label_total: dict[str, int] = {}
    for e in entries:
        key = f"{e['lang']}/{e['label']}"
        label_total[key] = label_total.get(key, 0) + 1

    rng = Random(int(_CFG["split_seed"]))
    order = sorted(members)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(members[g]))
    counts = {s: dict.fromkeys(label_total, 0) for s in ratios}
    assignment: dict[str, str] = {}
    for group in order:
        per_label: dict[str, int] = {}
        for e in members[group]:
            key = f"{e['lang']}/{e['label']}"
            per_label[key] = per_label.get(key, 0) + 1

        def need(split: str) -> float:
            return sum(
                n
                * (ratios[split] * label_total[lab] - counts[split][lab])
                / label_total[lab]
                for lab, n in per_label.items()
            )

        split = max(ratios, key=need)
        for lab, n in per_label.items():
            counts[split][lab] += n
        for e in members[group]:
            assignment[e["utt_id"]] = split
    return assignment


def _download_source(destination: Path) -> None:
    """Fetch the corpus from Kaggle and unpack it into ``destination``.

    Kaggle is the only distribution point, so the Kaggle CLI and an API token
    are required, as in ``egs3/jlcorpus/cls``.

    Args:
        destination: Directory to unpack into.

    Raises:
        RuntimeError: If the Kaggle CLI is missing.
        subprocess.CalledProcessError: If the download fails.
        FileNotFoundError: If Kaggle wrote no archive.
        zipfile.BadZipFile: If the archive cannot be extracted.
    """
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The `kaggle` command is required to download C-BESD, which is "
            "published on Kaggle. Install it with `pip install kaggle`, then put "
            "an API token from https://www.kaggle.com/settings/api in "
            "~/.kaggle/access_token or in $KAGGLE_API_TOKEN."
        )
    destination.mkdir(parents=True, exist_ok=True)
    dataset = str(_CFG["kaggle_dataset"])
    logger.info("Downloading %s from Kaggle", dataset)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(destination)],
        check=True,
    )

    archives = sorted(destination.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"Kaggle wrote no archive into {destination}")
    for archive in archives:
        _extract_zip(archive, destination)
        if _CFG.get("remove_archive", False):
            archive.unlink(missing_ok=True)


def _extract_zip(archive: Path, destination: Path) -> None:
    """Extract one zip archive into ``destination``.

    Only members starting with ``extract_prefix`` and ending with
    ``extract_suffix`` (both optional in ``dataset/config.yaml``) are written.
    Members are checked against the destination first: ``ZipFile.extractall``
    does not reject absolute paths or ``..`` components on its own.

    Args:
        archive: Archive to read.
        destination: Directory to extract into.

    Raises:
        ValueError: If a member would be written outside ``destination``.
        zipfile.BadZipFile: If the archive is invalid.
    """
    prefix = str(_CFG.get("extract_prefix") or "")
    suffix = str(_CFG.get("extract_suffix") or "").lower()
    logger.info("Extracting %s", archive.name)
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = [
            name
            for name in zf.namelist()
            if name.startswith(prefix) and name.lower().endswith(suffix)
        ]
        for name in members:
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError(f"{archive.name} escapes its destination: {name}")
        zf.extractall(path=destination, members=members)


class CbesdBuilder(DatasetBuilder):
    """Prepare C-BESD for the classification recipe.

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
        """Download and unpack the corpus unless it is already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the tree is still incomplete afterwards.
            RuntimeError: If the Kaggle CLI is missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("C-BESD already present; skipping download")
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
