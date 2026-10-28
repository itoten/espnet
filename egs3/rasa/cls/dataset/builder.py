"""Rasa dataset builder."""

from __future__ import annotations

import io
import json
import logging
import os
import random
import shutil
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from pathlib import Path
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the downloaded parquet files.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate download roots, most specific first.
    """
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the converted audio and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$RASA_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        return token_file.read_text().strip() or None
    return None


def _consent_error() -> PermissionError:
    return PermissionError(
        f"Access to {_CFG['hf_repo']} was refused. Sign in to Hugging Face, open "
        f"https://huggingface.co/datasets/{_CFG['hf_repo']} and agree to share "
        "your username and email with the authors, then make the account's "
        "token available (`huggingface-cli login` or $HF_TOKEN)."
    )


def _hf_request(url: str, headers: dict | None = None):
    headers = dict(headers or {})
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=120
        )
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            raise _consent_error() from err
        raise


def _list_parquet(language: str) -> list[tuple[str, int]]:
    """Return ``(path, size)`` of the parquet files of one language."""
    url = (
        f"https://huggingface.co/api/datasets/{_CFG['hf_repo']}/tree/main/"
        f"{language}?recursive=true"
    )
    with _hf_request(url) as response:
        entries = json.load(response)
    return sorted(
        (e["path"], int(e["size"])) for e in entries if e["path"].endswith(".parquet")
    )


def _fetch(path: str, size: int, target: Path, attempts: int = 5) -> None:
    """Download one repository file, resuming a partial download."""
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{_CFG['hf_repo']}/resolve/main/{path}"
    for _ in range(attempts):
        have = target.stat().st_size if target.is_file() else 0
        if have == size:
            return
        try:
            with _hf_request(url, {"Range": f"bytes={have}-"} if have else {}) as r:
                with target.open("ab" if have else "wb") as fh:
                    shutil.copyfileobj(r, fh, 1 << 20)
        except PermissionError:
            raise
        except OSError as err:
            logger.warning("Retrying %s: %s", path, err)
    if target.stat().st_size != size:
        raise ValueError(f"Incomplete download: {path}")


def _convert_file(job: tuple[str, int, str, str]) -> int:
    """Download one parquet file, write its utterances as 16 kHz wav, delete it.

    The rows are recorded in ``progress/<path>.tsv``, written last, so that a
    finished file is skipped on a re-run.

    Returns:
        int: Number of utterances in the file.
    """
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr

    path, size, download_root, data_root = job
    progress = Path(data_root) / _CFG["progress_subdir"] / (path + ".tsv")
    if progress.is_file():
        return sum(1 for _ in progress.open(encoding="utf-8"))
    parquet = Path(download_root) / path
    _fetch(path, size, parquet)

    language = path.split("/", 1)[0]
    split = Path(path).name.split("-", 1)[0]
    wav_dir = Path(data_root) / _CFG["wav_subdir"] / language
    wav_dir.mkdir(parents=True, exist_ok=True)
    rate = int(_CFG["sampling_rate"])
    lines = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            data, source_rate = sf.read(
                io.BytesIO(row["audio"]["bytes"]), dtype="float32"
            )
            if data.ndim > 1:
                data = data.mean(axis=1)
            if source_rate != rate:
                data = soxr.resample(data, source_rate, rate)
            wav = wav_dir / f"{row['filename']}.wav"
            sf.write(wav.with_suffix(".part.wav"), data, rate, subtype="PCM_16")
            wav.with_suffix(".part.wav").replace(wav)
            lines.append(
                "\t".join(
                    [
                        f"rasa-{row['filename']}",
                        str(wav.resolve()),
                        row["style"],
                        language,
                        row["gender"],
                        split,
                        f"{len(data) / rate:.3f}",
                    ]
                )
            )
    progress.parent.mkdir(parents=True, exist_ok=True)
    part = progress.with_suffix(".tsv.part")
    part.write_text("\n".join(lines) + "\n", encoding="utf-8")
    part.replace(progress)
    parquet.unlink()
    return len(lines)


def _label(style: str) -> str:
    return style.strip().lower().replace(" ", "_")


def read_entries(data_root: Path) -> list[dict[str, str]]:
    """Read the progress files and assign each utterance to a split.

    Returns:
        list[dict[str, str]]: Keys ``utt_id``, ``wav``, ``label``, ``language``,
        ``gender`` and ``split``. The published ``test`` split is kept; a random
        ``valid_ratio`` of the published ``train`` utterances of each
        (language, gender, label) group becomes ``valid``.

    A filename that appears twice (same audio, different style) is left out
    in both places.
    """
    entries = []
    for path in sorted((data_root / _CFG["progress_subdir"]).glob("*/*.tsv")):
        for line in path.read_text(encoding="utf-8").splitlines():
            utt_id, wav, style, language, gender, split, _ = line.split("\t")
            entries.append(
                {
                    "utt_id": utt_id,
                    "wav": wav,
                    "label": _label(style),
                    "language": language,
                    "gender": gender,
                    "split": split,
                }
            )
    counts = Counter(e["utt_id"] for e in entries)
    clashing = {i for i, n in counts.items() if n > 1}
    if clashing:
        # A reused filename holds one recording under two texts and styles.
        logger.warning("Leaving out %d filenames used twice", len(clashing))
        entries = [e for e in entries if e["utt_id"] not in clashing]
    groups = defaultdict(list)
    for entry in entries:
        if entry["split"] == "train":
            groups[(entry["language"], entry["gender"], entry["label"])].append(entry)
    rng = random.Random(int(_CFG["split_seed"]))
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda e: e["utt_id"])
        rng.shuffle(members)
        for entry in members[: round(len(members) * float(_CFG["valid_ratio"]))]:
            entry["split"] = "valid"
    if _CFG["exclude_duplicates"]:
        drop = {str(i) for i in _CFG["duplicate_ids"]}
        missing = drop - {e["utt_id"] for e in entries}
        if missing:
            raise ValueError(f"`duplicate_ids` not found: {sorted(missing)[:5]}")
        entries = [e for e in entries if e["utt_id"] not in drop]
    return entries


class RasaBuilder(DatasetBuilder):
    """Prepare Rasa for the classification recipe.

    `prepare_source` goes through the 22 languages one at a time: it downloads
    each parquet file, converts its utterances to 16 kHz mono wav and deletes the
    file, so that the 388 GB of the Hugging Face dataset is never on disk at
    once. `build()` writes one manifest per split.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether every parquet file has been converted."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        return (data_root / _CFG["progress_subdir"] / ".complete").is_file()

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Download and convert the files that are not converted yet.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Directory the parquet files are downloaded to
                (temporarily).

        Raises:
            PermissionError: If the account has not agreed to the dataset's terms.
        """
        recipe_root = Path(recipe_dir).resolve()
        data_root = resolve_data_root(recipe_root)
        download_root = next(iter_source_candidates(recipe_root, source_dir))
        marker = data_root / _CFG["progress_subdir"] / ".complete"
        if marker.is_file():
            logger.info("Rasa already converted; skipping download")
            return
        with ProcessPoolExecutor(int(_CFG["download_workers"])) as pool:
            for language in _CFG["languages"]:
                jobs = [
                    (path, size, str(download_root), str(data_root))
                    for path, size in _list_parquet(language)
                ]
                count = sum(pool.map(_convert_file, jobs))
                logger.info(
                    "%s: %d utterance(s) in %d file(s)", language, count, len(jobs)
                )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("done\n")

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
        """Write one TSV manifest per split from the converted utterances."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        for entry in read_entries(data_root):
            rows[entry["split"]].append((entry["utt_id"], entry["wav"], entry["label"]))
        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(rows[split]), manifest)
