"""E-SERAVD dataset builder (manually obtained Kaggle archive)."""

from __future__ import annotations

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
        "Complete E-SERAVD audio not found. Checked:\n  "
        + "\n  ".join(checked)
        + "\nLet `prepare_source` download it with the Kaggle CLI (an API token "
        "in ~/.kaggle/access_token or $KAGGLE_API_TOKEN), or download the "
        f"archive from {_CFG['kaggle_url']} (a free Kaggle "
        "account is needed), extract it, and set "
        f"${_CFG['source_env_var']} to the extracted directory."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$ESERAVD_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


# `02-05-AADC-T-0001.wav`: 02 = audio-only, 05 = emotion code, AADC = source
# film, T = tested (agreed) clip. Only the files of one release folder are used.
FILENAME_RE = re.compile(
    r"^(?P<modality>\d\d)-(?P<code>\d\d)-(?P<film>[A-Z0-9]+)-T-(?P<clip>\d+)\.wav$"
)


def _iter_wavs(root: Path) -> Iterator[Path]:
    return iter(sorted((root / str(_CFG["release_dir"])).rglob("*.wav")))


def read_entries(source_root: Path) -> list[dict[str, str]]:
    """Collect the recordings, taking the label from the emotion code in the name.

    Files named in ``exclude_files`` are skipped.

    Args:
        source_root: Directory holding the release folder (``release_dir``).

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``source``, ``group`` (the source
        film) and ``label``.

    Raises:
        ValueError: If a name does not parse, the emotion code is unknown or
            disagrees with the folder, two files share a name, or an excluded
            name is not found.
        FileNotFoundError: If no recording is found.
    """
    excluded = {str(n) for n in _CFG["exclude_files"]}
    entries = []
    seen: dict[str, Path] = {}
    for wav in _iter_wavs(source_root):
        fields = FILENAME_RE.match(wav.name)
        if fields is None:
            raise ValueError(f"Unexpected file name: {wav}")
        if not wav.parent.parent.name.startswith(fields["code"]):
            raise ValueError(f"Emotion code and folder disagree: {wav}")
        if fields["code"] not in _EMOTION_MAP:
            raise ValueError(
                f"Unknown emotion code {fields['code']!r} in {wav}. "
                f"Known: {sorted(_EMOTION_MAP)}"
            )
        if wav.stem in seen:
            raise ValueError(f"Duplicate recording name: {wav} and {seen[wav.stem]}")
        seen[wav.stem] = wav
        if wav.name in excluded:
            excluded.discard(wav.name)
            continue
        entries.append(
            {
                "utt_id": f"eseravd-{wav.stem}",
                "source": str(wav),
                "group": fields["film"],
                "label": _EMOTION_MAP[fields["code"]],
            }
        )
    if excluded:
        raise ValueError(f"`exclude_files` names not found: {sorted(excluded)}")
    if not entries:
        raise FileNotFoundError(f"No E-SERAVD audio under {source_root}")
    logger.info(
        "Read %d utterance(s) from %d film(s)",
        len(entries),
        len({e["group"] for e in entries}),
    )
    return entries


def assign_splits(entries: list[dict[str, str]]) -> dict[str, str]:
    """Hold out whole films: ``n_test_groups``, ``n_valid_groups``, rest train.

    Films are shuffled with ``split_seed``, so every split has films the
    others never see.

    Args:
        entries: Rows from :func:`read_entries`.

    Returns:
        dict[str, str]: Utterance id to split name.

    Raises:
        ValueError: If fewer films exist than the two held-out splits need
            plus one for training.
    """
    n_test = int(_CFG["n_test_groups"])
    n_valid = int(_CFG["n_valid_groups"])
    groups = sorted({e["group"] for e in entries})
    if len(groups) < n_test + n_valid + 1:
        raise ValueError(f"Only {len(groups)} films for {n_test}+{n_valid}+train")
    Random(int(_CFG["split_seed"])).shuffle(groups)
    split_of = {g: "test" for g in groups[:n_test]}
    split_of.update({g: "valid" for g in groups[n_test : n_test + n_valid]})
    return {
        e["utt_id"]: split_of.get(e["group"], "train")
        for e in sorted(entries, key=lambda e: e["utt_id"])
    }


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
            "The `kaggle` command is required to download E-SERAVD, which is "
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


class EseravdBuilder(DatasetBuilder):
    """Prepare E-SERAVD for the classification recipe.

    `build()` resamples every recording to 16 kHz mono and writes
    one manifest per split, drawn as a film-held-out split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a E-SERAVD source tree is reachable."""
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
            logger.info("E-SERAVD already present; skipping download")
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
