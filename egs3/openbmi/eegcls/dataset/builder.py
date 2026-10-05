"""OpenBMI dataset builder."""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()

# Fixed by the corpus: 54 subjects, 2 sessions, and each `.mat` file holds a
# "train" and a "test" phase of 100 trials.
_NUM_SUBJECTS = 54
_SESSIONS = (1, 2)
_PHASES = ("train", "test")


def source_relpath(subject: int, session: int) -> str:
    """Return the path of one `.mat` file relative to the corpus root.

    The folder name is not zero-padded, the file name is, e.g.
    ``session1/s7/sess01_subj07_EEG_MI.mat``.
    """
    return f"session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat"


def resolve_source_root(source_dir: str | Path | None = None) -> Path:
    """Return the directory that holds `session1/` and `session2/`.

    Args:
        source_dir: Explicit location, used before the environment variable.

    Returns:
        Path: Corpus root.

    Raises:
        FileNotFoundError: If neither a location nor the environment variable
            is given.
    """
    if source_dir:
        return Path(source_dir).expanduser()
    env_value = os.environ.get(_CFG["source_env_var"])
    if env_value:
        return Path(env_value).expanduser()
    raise FileNotFoundError(
        f"OpenBMI location unknown. Set ${_CFG['source_env_var']} to the "
        "directory holding `session1/` and `session2/`."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the epochs and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$OPENBMI_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(_CFG["output_env_var"])
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def missing_source_files(source_root: Path) -> list[Path]:
    """Return the `.mat` files that are absent under ``source_root``."""
    missing = []
    for subject in range(1, _NUM_SUBJECTS + 1):
        for session in _SESSIONS:
            path = source_root / source_relpath(subject, session)
            if not path.is_file():
                missing.append(path)
    return missing


def assign_split(subject: int) -> str:
    """Return the split a subject belongs to: ``test``, ``valid`` or ``train``."""
    if subject in _CFG["test_subjects"]:
        return "test"
    if subject in _CFG["valid_subjects"]:
        return "valid"
    return "train"


def make_utt_id(subject: int, session: int, phase: str, trial: int) -> str:
    """Return the utterance id of one trial, e.g. ``s07_sess2_test_042``."""
    return f"s{subject:02d}_sess{session}_{phase}_{trial:03d}"


def read_trials(mat_path: Path, phase: str) -> tuple[np.ndarray, list[str]]:
    """Read one phase of one file as decimated trials with their labels.

    Args:
        mat_path: A `.mat` file of the corpus.
        phase: ``train`` or ``test``.

    Returns:
        tuple: Trials of shape ``(trials, time, channels)`` as float32, and one
        label name per trial.
    """
    struct_name = f"EEG_MI_{phase}"
    struct = loadmat(str(mat_path), variable_names=[struct_name])[struct_name][0, 0]

    # `smt` is (time, trials, channels), already cut to 0-4 s after the cue.
    # `y_dec` is 1 or 2 per trial; `class` maps it to a name: [['1', 'right'], ...].
    signal = struct["smt"]
    label_ids = struct["y_dec"].ravel()
    names = {int(row[0].item()): str(row[1].item()) for row in struct["class"]}
    labels = [names[int(i)] for i in label_ids]

    # Decimate along time, then put trials first: (trials, time, channels).
    signal = decimate(signal, _CFG["decimate_factor"], axis=0)
    trials = np.ascontiguousarray(signal.transpose(1, 0, 2), dtype=np.float32)
    return trials, labels


def _download_missing(source_root: Path) -> None:
    """Fetch the `.mat` files that are absent."""
    base_url = _CFG["base_url"].rstrip("/")
    for path in missing_source_files(source_root):
        relpath = path.relative_to(source_root).as_posix()
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        download_url(f"{base_url}/{relpath}", part, logger=logger)
        part.rename(path)


class OpenBMIBuilder(DatasetBuilder):
    """Prepare OpenBMI for the EEG classification recipe.

    `build()` decimates every trial, saves it as one `.npy` file and writes one
    manifest per split.
    """

    def is_source_prepared(
        self, recipe_dir: str | Path, source_dir: str | Path | None = None, **_kwargs
    ) -> bool:
        """Check whether all `.mat` files are present."""
        try:
            return not missing_source_files(resolve_source_root(source_dir))
        except FileNotFoundError:
            return False

    def prepare_source(
        self, recipe_dir: str | Path, source_dir: str | Path | None = None, **_kwargs
    ) -> None:
        """Make all `.mat` files available, downloading the missing ones.

        Args:
            recipe_dir: Recipe directory.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the location is unknown, or files are still
                missing afterwards.
        """
        source_root = resolve_source_root(source_dir)
        if not missing_source_files(source_root):
            logger.info("OpenBMI already present; skipping download")
            return
        _download_missing(source_root)
        still_missing = missing_source_files(source_root)
        if still_missing:
            raise FileNotFoundError(f"Still missing after download: {still_missing[0]}")

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether every split manifest already exists."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        manifests = _CFG["manifest_paths"]
        return all((data_root / manifests[split]).is_file() for split in manifests)

    def build(
        self, recipe_dir: str | Path, source_dir: str | Path | None = None, **_kwargs
    ) -> None:
        """Write the trials as `.npy` files and one TSV manifest per split.

        Args:
            recipe_dir: Recipe directory.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If a `.mat` file is missing.
        """
        source_root = resolve_source_root(source_dir)
        missing = missing_source_files(source_root)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} file(s) absent under {source_root}, starting "
                f"with {missing[0]}. Let `prepare_source` download them or fix "
                f"${_CFG['source_env_var']}."
            )
        data_root = resolve_data_root(Path(recipe_dir).resolve())

        rows = {"train": [], "valid": [], "test": []}
        for subject in range(1, _NUM_SUBJECTS + 1):
            split = assign_split(subject)
            epoch_dir = data_root / "epochs" / f"s{subject:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            for session in _SESSIONS:
                mat_path = source_root / source_relpath(subject, session)
                for phase in _PHASES:
                    trials, labels = read_trials(mat_path, phase)
                    for index, (trial, label) in enumerate(zip(trials, labels)):
                        utt_id = make_utt_id(subject, session, phase, index)
                        npy_path = epoch_dir / f"{utt_id}.npy"
                        np.save(npy_path, trial)
                        rows[split].append((utt_id, str(npy_path), label))
            logger.info("subject %d/%d done (%s)", subject, _NUM_SUBJECTS, split)

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / relpath
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Part file and rename, so an interrupted build cannot leave a
            # truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, npy_path, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{npy_path}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
