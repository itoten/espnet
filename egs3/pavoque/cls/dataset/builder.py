"""PAVOQUE dataset builder."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import wave
from importlib import resources
from pathlib import Path
from random import Random
from typing import Iterator

import yaml

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url

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
    """Yield the directories that may hold the corpus source tree, in order.

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


def _source_names() -> list[str]:
    return [
        f"pavoque-{style}.{ext}" for style in _CFG["styles"] for ext in ("flac", "yaml")
    ]


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate holding every PAVOQUE FLAC and YAML file.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: Directory with ``pavoque-<style>.flac`` and ``.yaml`` per style.

    Raises:
        FileNotFoundError: If no candidate holds all of them.
    """
    names = _source_names()
    checked = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        if all((candidate / name).is_file() for name in names):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "Complete PAVOQUE files not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to a directory holding "
        + ", ".join(names)
        + ", or let `prepare_source` download them from the GitHub release."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$PAVOQUE_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _download_source(destination: Path) -> None:
    """Fetch the FLAC and YAML file of every style.

    Args:
        destination: Directory to download into.

    Raises:
        urllib.error.URLError: If a download fails.
    """
    destination.mkdir(parents=True, exist_ok=True)
    base_url = str(_CFG["base_url"]).rstrip("/")
    for name in _source_names():
        target = destination / name
        if target.is_file():
            continue
        part = target.with_suffix(target.suffix + ".part")
        download_url(f"{base_url}/{name}", part, logger=logger)
        part.rename(target)


def read_entries(source_root: Path) -> list[dict]:
    """Collect the utterances listed in the per-style YAML files.

    Args:
        source_root: Directory with the downloaded files.

    Returns:
        list[dict]: Keys ``utt_id``, ``style``, ``start``, ``end`` (seconds in
        the style's FLAC) and ``label``.

    Raises:
        ValueError: If an utterance's style disagrees with its file, is
            unknown, or two utterances share an id.
        FileNotFoundError: If no utterance is found.
    """
    entries = []
    seen = set()
    for style in _CFG["styles"]:
        with (source_root / f"pavoque-{style}.yaml").open(encoding="utf-8") as fh:
            items = yaml.safe_load(fh)
        for item in items:
            if item["style"] != style:
                raise ValueError(
                    f"{item['prompt']} in pavoque-{style}.yaml has style "
                    f"{item['style']!r}"
                )
            if style not in _EMOTION_MAP:
                raise ValueError(
                    f"Unknown style {style!r}. Known: {sorted(_EMOTION_MAP)}"
                )
            utt_id = f"pavoque-{style}-{item['prompt']}"
            if utt_id in seen:
                raise ValueError(f"Duplicate utterance id: {utt_id}")
            seen.add(utt_id)
            entries.append(
                {
                    "utt_id": utt_id,
                    "style": style,
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "label": _EMOTION_MAP[style],
                }
            )
    if not entries:
        raise FileNotFoundError(f"No PAVOQUE utterances under {source_root}")
    logger.info("Read %d utterance(s)", len(entries))
    return entries


def cut_utterances(
    long_wav: Path, cuts: list[tuple[float, float, Path]], tolerance: float = 0.05
) -> None:
    """Write each ``(start, end, path)`` slice of a long WAV file.

    Args:
        long_wav: 16-bit PCM WAV file holding the whole recording.
        cuts: Start and end in seconds, and the file to write.
        tolerance: Seconds an ``end`` may run past the recording.

    Raises:
        ValueError: If a slice starts after, or ends beyond, the recording.
    """
    with wave.open(str(long_wav), "rb") as src:
        rate = src.getframerate()
        total = src.getnframes()
        params = src.getparams()
        for start, end, path in cuts:
            first = round(start * rate)
            last = round(end * rate)
            if first >= total or last > total + tolerance * rate:
                raise ValueError(f"{path.name}: {start}-{end}s is beyond the file")
            src.setpos(first)
            frames = src.readframes(min(last, total) - first)
            part = path.with_suffix(path.suffix + ".part")
            with wave.open(str(part), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(frames)
            part.replace(path)


def assign_splits(entries: list[dict]) -> dict[str, str]:
    """Split the utterances at random within each label.

    Each label's utterances are shuffled with ``split_seed`` and cut by
    ``train_ratio``/``valid_ratio``/``test_ratio``. The corpus has one speaker.

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


class PavoqueBuilder(DatasetBuilder):
    """Prepare PAVOQUE for the classification recipe.

    `build()` converts each style's FLAC to 16 kHz mono, cuts it into
    utterances and writes one manifest
    per split, drawn as a label-stratified random split (see
    :func:`assign_splits`).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a PAVOQUE source tree is reachable."""
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
            logger.info("PAVOQUE already present; skipping download")
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
        """Cut the utterances out of the recordings and write a TSV manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit corpus location.

        Raises:
            FileNotFoundError: If the corpus or ``ffmpeg`` cannot be found.
            subprocess.CalledProcessError: If a resample fails.
            ValueError: If a YAML time range lies outside its FLAC.
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
        for style in _CFG["styles"]:
            style_entries = [e for e in entries if e["style"] == style]
            todo = [
                e
                for e in style_entries
                if not (wav_dir / f"{e['utt_id']}.wav").is_file()
            ]
            if todo:
                long_wav = data_root / f"{style}_long.wav.part"
                subprocess.run(
                    [
                        ffmpeg,
                        "-i",
                        str(source_root / f"pavoque-{style}.flac"),
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
                        str(long_wav),
                    ],
                    check=True,
                )
                try:
                    cut_utterances(
                        long_wav,
                        [
                            (e["start"], e["end"], wav_dir / f"{e['utt_id']}.wav")
                            for e in todo
                        ],
                    )
                finally:
                    long_wav.unlink(missing_ok=True)
            for entry in style_entries:
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
