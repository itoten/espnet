"""BEAT dataset builder (English emotional speech, cut into segments, 8 classes)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
_EMOTION_MAP = {str(k): str(v) for k, v in _CFG["emotion_map"].items()}
_STEM_RE = re.compile(str(_CFG["stem_re"]))
_SPLIT_OF_SPEAKER = {
    int(speaker): str(split)
    for split, speakers in _CFG["split_speakers"].items()
    for speaker in speakers
}


# ---------------------------------------------------------------------------
# TextGrid parsing and segmentation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """One interval of a TextGrid interval tier."""

    xmin: float
    xmax: float
    text: str


@dataclass(frozen=True)
class Segment:
    """A cut of a recording: start/end in seconds and the words inside."""

    start: float
    end: float
    words: tuple[str, ...]


_KEY_VALUE_RE = re.compile(r'^\s*(\w+)\s*=\s*(.*?)\s*$')
_INTERVAL_HEAD_RE = re.compile(r"^\s*intervals\s*\[\d+\]\s*:?\s*$")
_ITEM_HEAD_RE = re.compile(r"^\s*item\s*\[\d+\]\s*:?\s*$")


def _read_text(path: Path) -> str:
    """Read a TextGrid as text; Praat writes UTF-8 or UTF-16 (with BOM)."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def _unquote(value: str) -> str:
    """Return a Praat string value without its quotes (``""`` is one ``"``)."""
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value


def parse_textgrid(text: str) -> dict[str, list[Interval]]:
    """Parse the interval tiers of a Praat TextGrid in the long text format.

    Only ``IntervalTier`` items are returned; point tiers are skipped. Each
    interval needs ``xmin``, ``xmax`` and ``text`` lines, in any order.

    Args:
        text: Content of the ``.TextGrid`` file.

    Returns:
        dict[str, list[Interval]]: Tier name to its intervals, in file order.

    Raises:
        ValueError: If the text is not a long-format TextGrid or an interval
            is incomplete.
    """
    if 'Object class = "TextGrid"' not in text:
        raise ValueError("Not a TextGrid in the long text format")
    tiers: dict[str, list[Interval]] = {}
    tier_class = tier_name = None
    current: dict[str, str] | None = None

    def flush() -> None:
        if current is None or tier_class != "IntervalTier" or tier_name is None:
            return
        try:
            interval = Interval(
                float(current["xmin"]), float(current["xmax"]), current["text"]
            )
        except KeyError as exc:
            raise ValueError(
                f"Incomplete interval in tier {tier_name!r}: {current}"
            ) from exc
        tiers[tier_name].append(interval)

    for line in text.splitlines():
        if _ITEM_HEAD_RE.match(line):
            flush()
            current = None
            tier_class = tier_name = None
            continue
        if _INTERVAL_HEAD_RE.match(line):
            flush()
            current = {}
            continue
        match = _KEY_VALUE_RE.match(line)
        if match is None:
            continue
        key, value = match.group(1), match.group(2)
        if current is not None:
            if key in ("xmin", "xmax", "text"):
                current[key] = _unquote(value) if key == "text" else value
        elif key == "class":
            tier_class = _unquote(value)
        elif key == "name" and tier_class is not None:
            tier_name = _unquote(value)
            if tier_class == "IntervalTier":
                tiers.setdefault(tier_name, [])
    flush()
    return tiers


def _split_long(
    words: list[Interval], max_segment: float
) -> list[list[Interval]]:
    """Split a run of words at its longest internal pause until each fits."""
    if len(words) < 2 or words[-1].xmax - words[0].xmin <= max_segment:
        return [words]
    gaps = [words[i + 1].xmin - words[i].xmax for i in range(len(words) - 1)]
    cut = max(range(len(gaps)), key=lambda i: (gaps[i], -i)) + 1
    return _split_long(words[:cut], max_segment) + _split_long(
        words[cut:], max_segment
    )


def segment_words(
    intervals: list[Interval],
    *,
    silence_labels: set[str],
    pause_threshold: float,
    max_segment: float,
    min_segment: float,
    padding: float,
    total: float | None = None,
) -> tuple[list[Segment], int]:
    """Group the words of one tier into segments that break at pauses.

    Consecutive words stay together until the gap between two words is at
    least ``pause_threshold``. A group longer than ``max_segment`` is split at
    its longest internal gap, repeatedly. Groups shorter than ``min_segment``
    (first word start to last word end) are dropped. Each kept group is
    widened by ``padding`` on both sides, but not past the middle of the gap
    to the neighbouring word, nor below 0 or above ``total``.

    Args:
        intervals: The tier's intervals in time order.
        silence_labels: Interval texts that mean "no word".
        pause_threshold: Minimum gap (s) that ends a segment.
        max_segment: Maximum length (s) of a segment before padding.
        min_segment: Minimum length (s) of a segment before padding.
        padding: Context (s) added on each side.
        total: End of the recording (s); defaults to the last interval's end.

    Returns:
        tuple[list[Segment], int]: Kept segments in time order, and the number
        of groups dropped for being too short (or a single word longer than
        ``max_segment``).
    """
    words = sorted(
        (iv for iv in intervals if iv.text.strip() not in silence_labels),
        key=lambda iv: (iv.xmin, iv.xmax),
    )
    if total is None:
        total = max((iv.xmax for iv in intervals), default=0.0)
    if not words:
        return [], 0

    groups: list[list[Interval]] = [[words[0]]]
    for word in words[1:]:
        if word.xmin - groups[-1][-1].xmax >= pause_threshold:
            groups.append([word])
        else:
            groups[-1].append(word)
    pieces = [p for g in groups for p in _split_long(g, max_segment)]

    segments: list[Segment] = []
    dropped = 0
    for i, piece in enumerate(pieces):
        first, last = piece[0].xmin, piece[-1].xmax
        length = last - first
        if length < min_segment or length > max_segment:
            dropped += 1
            continue
        prev_end = pieces[i - 1][-1].xmax if i > 0 else 0.0
        next_start = pieces[i + 1][0].xmin if i + 1 < len(pieces) else total
        start = max(first - padding, (prev_end + first) / 2 if i > 0 else 0.0)
        end = min(
            last + padding, (last + next_start) / 2 if i + 1 < len(pieces) else total
        )
        segments.append(
            Segment(
                round(max(start, 0.0), 3),
                round(min(end, total), 3),
                tuple(w.text.strip() for w in piece),
            )
        )
    return segments, dropped


# ---------------------------------------------------------------------------
# Source and output locations
# ---------------------------------------------------------------------------


def iter_source_candidates(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Iterator[Path]:
    """Yield the directories that may hold the BEAT snapshot, in order.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Yields:
        Path: Candidate source roots, most specific first.
    """
    if source_dir:
        yield Path(source_dir).expanduser()
    env_value = os.environ.get(str(_CFG["source_env_var"]))
    if env_value:
        yield Path(env_value).expanduser()
    yield recipe_root / _CFG["dataset_path"]


def _release_dir(candidate: Path) -> Path:
    """Return the folder holding the per-speaker folders under ``candidate``."""
    inner = candidate / str(_CFG["release_subdir"])
    return inner if inner.is_dir() else candidate


def find_recordings(release_dir: Path) -> tuple[list[dict[str, str]], Counter]:
    """List the recordings that have a wav, a TextGrid and a csv file.

    Args:
        release_dir: Folder holding ``<speaker id>/<stem>.*``.

    Returns:
        tuple[list[dict[str, str]], Counter]: One dict per complete recording
        (keys ``stem``, ``speaker``, ``type``, ``dir``), sorted by stem, and
        counts of skipped files by reason.
    """
    recordings = []
    skipped: Counter[str] = Counter()
    if not release_dir.is_dir():
        return recordings, skipped
    for folder in sorted(p for p in release_dir.iterdir() if p.is_dir()):
        for wav in sorted(folder.glob("*.wav")):
            fields = _STEM_RE.match(wav.stem)
            if fields is None:
                skipped["unexpected file name"] += 1
                continue
            if fields["speaker"] != folder.name:
                skipped["in another speaker's folder"] += 1
                continue
            if not all(
                wav.with_suffix(ext).is_file() for ext in (".TextGrid", ".csv")
            ):
                skipped["no TextGrid or csv"] += 1
                continue
            recordings.append(
                {
                    "stem": wav.stem,
                    "speaker": fields["speaker"],
                    "type": fields["type"],
                    "dir": str(folder),
                }
            )
    recordings.sort(key=lambda r: r["stem"])
    return recordings, skipped


def resolve_source_root(
    recipe_root: Path, source_dir: str | Path | None = None
) -> Path:
    """Return the first candidate with every expected recording complete.

    Args:
        recipe_root: Resolved recipe directory.
        source_dir: Explicit location, tried first when given.

    Returns:
        Path: The folder holding the per-speaker folders.

    Raises:
        FileNotFoundError: If no candidate is complete.
    """
    checked = []
    expected = int(_CFG["expected_recordings"])
    for candidate in iter_source_candidates(recipe_root, source_dir):
        release_dir = _release_dir(candidate)
        found = len(find_recordings(release_dir)[0])
        if found >= expected:
            return release_dir
        checked.append(f"{release_dir} ({found} of {expected} recordings)")
    raise FileNotFoundError(
        "Complete BEAT source not found. Checked:\n  "
        + "\n  ".join(checked)
        + f"\nSet ${_CFG['source_env_var']} to the Hugging Face snapshot "
        f"(or its `{_CFG['release_subdir']}` folder), or let `prepare_source` "
        "download it."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory the segments and manifests are written into.

    Args:
        recipe_root: Resolved recipe directory.

    Returns:
        Path: ``$BEAT_OUTPUT`` when set, else ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / _CFG["data_path"]


def read_label(csv_path: Path) -> str:
    """Return the recording's label from its ``.csv`` (first field).

    Raises:
        ValueError: If the file is empty, holds different labels, or the label
            is not in ``emotion_map``.
    """
    labels = {
        line.split(",", 1)[0].strip()
        for line in csv_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(labels) != 1:
        raise ValueError(f"Expected one label in {csv_path}, found {sorted(labels)}")
    label = labels.pop()
    if label not in _EMOTION_MAP:
        raise ValueError(
            f"Unknown label {label!r} in {csv_path}. Known: {sorted(_EMOTION_MAP)}"
        )
    return _EMOTION_MAP[label]


def segment_recording(textgrid_path: Path) -> tuple[list[Segment], int]:
    """Cut one recording into segments with the settings of ``config.yaml``."""
    tiers = parse_textgrid(_read_text(textgrid_path))
    tier = str(_CFG["textgrid_tier"])
    if tier not in tiers:
        raise ValueError(f"No tier {tier!r} in {textgrid_path}: {sorted(tiers)}")
    return segment_words(
        tiers[tier],
        silence_labels={str(s) for s in _CFG["silence_labels"]},
        pause_threshold=float(_CFG["pause_threshold"]),
        max_segment=float(_CFG["max_segment"]),
        min_segment=float(_CFG["min_segment"]),
        padding=float(_CFG["padding"]),
    )


def _cut(
    ffmpeg: str, source: Path, start: float, end: float, wav: Path, rate: str
) -> None:
    """Write ``source[start:end]`` as 16 kHz mono 16-bit PCM."""
    part = wav.with_suffix(wav.suffix + ".part")
    subprocess.run(
        [
            ffmpeg, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(source), "-ac", "1", "-ar", rate, "-c:a", "pcm_s16le",
            "-f", "wav", "-y", "-hide_banner", "-loglevel", "error", str(part),
        ],
        check=True,
    )
    part.replace(wav)


class BeatBuilder(DatasetBuilder):
    """Prepare BEAT for the classification recipe.

    `prepare_source` downloads the audio, the TextGrid alignments and the
    per-recording label files from Hugging Face at a pinned revision.
    `build()` cuts every recording into segments along its word alignment
    (see :func:`segment_words`), writes each as 16 kHz mono 16-bit PCM and
    writes one manifest per split by speaker id. Each segment inherits its
    recording's label.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete BEAT source is reachable."""
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
        """Download the wav, TextGrid and csv files unless already available.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location, tried before the environment
                variable and the recipe-local download directory.

        Raises:
            FileNotFoundError: If the source is still incomplete afterwards.
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir, source_dir):
            logger.info("BEAT already present; skipping download")
            return

        from huggingface_hub import snapshot_download

        destination = next(iter_source_candidates(recipe_root, source_dir))
        destination.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s to %s", _CFG["hf_repo"], destination)
        snapshot_download(
            repo_id=str(_CFG["hf_repo"]),
            repo_type="dataset",
            revision=str(_CFG["hf_revision"]),
            allow_patterns=[str(p) for p in _CFG["allow_patterns"]],
            local_dir=str(destination),
        )
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
        num_workers: int = 8,
        **_kwargs,
    ) -> None:
        """Cut the recordings into segments and write one manifest per split.

        Args:
            recipe_dir: Recipe directory, used to resolve relative locations.
            source_dir: Explicit source location.
            num_workers: ffmpeg processes run at the same time.

        Raises:
            FileNotFoundError: If the source or ``ffmpeg`` cannot be found.
            ValueError: If a label is unknown, a speaker is in no split, or a
                TextGrid cannot be parsed.
            subprocess.CalledProcessError: If a cut fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        release_dir = resolve_source_root(recipe_root, source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required to cut the recordings")

        wav_dir = data_root / str(_CFG["wav_subdir"])
        wav_dir.mkdir(parents=True, exist_ok=True)
        rate = str(_CFG["sampling_rate"])
        types = {str(t) for t in _CFG["recording_types"]}

        recordings, skipped = find_recordings(release_dir)
        for reason, count in sorted(skipped.items()):
            logger.warning("Skipped %d wav file(s): %s", count, reason)

        rows: dict[str, list[tuple[str, str, str]]] = {
            split: [] for split in _CFG["manifest_paths"]
        }
        segment_rows: list[str] = []
        jobs: list[tuple[Path, float, float, Path]] = []
        excluded_types: Counter[str] = Counter()
        too_short = 0
        empty = 0
        for rec in recordings:
            if rec["type"] not in types:
                excluded_types[rec["type"]] += 1
                continue
            split = _SPLIT_OF_SPEAKER.get(int(rec["speaker"]))
            if split is None:
                raise ValueError(f"Speaker {rec['speaker']} is in no split")
            folder = Path(rec["dir"])
            stem = rec["stem"]
            label = read_label(folder / f"{stem}.csv")
            segments, dropped = segment_recording(folder / f"{stem}.TextGrid")
            too_short += dropped
            if not segments:
                empty += 1
                logger.warning("No segment kept from %s", stem)
            for n, seg in enumerate(segments):
                utt_id = f"beat-{stem}_{n:03d}"
                wav = wav_dir / f"{utt_id}.wav"
                if not wav.is_file():
                    jobs.append((folder / f"{stem}.wav", seg.start, seg.end, wav))
                rows[split].append((utt_id, str(wav.resolve()), label))
                segment_rows.append(
                    f"{utt_id}\t{stem}\t{seg.start:.3f}\t{seg.end:.3f}\t{label}"
                    f"\t{' '.join(seg.words)}"
                )

        for rec_type, count in sorted(excluded_types.items()):
            logger.info("Left out %d recording(s) of type %s", count, rec_type)
        logger.info(
            "Dropped %d word group(s) outside %s-%s s; %d recording(s) gave no "
            "segment",
            too_short, _CFG["min_segment"], _CFG["max_segment"], empty,
        )
        logger.info("Cutting %d segment(s) with %d worker(s)", len(jobs), num_workers)
        with ThreadPoolExecutor(max_workers=int(num_workers)) as pool:
            list(pool.map(lambda j: _cut(ffmpeg, j[0], j[1], j[2], j[3], rate), jobs))

        segments_file = data_root / str(_CFG["segments_path"])
        part = segments_file.with_suffix(segments_file.suffix + ".part")
        part.write_text(
            "\n".join(sorted(segment_rows)) + "\n", encoding="utf-8"
        )
        part.replace(segments_file)

        for split, relpath in _CFG["manifest_paths"].items():
            manifest = data_root / str(relpath)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(rows[split]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            counts = Counter(label for _, _, label in rows[split])
            logger.info(
                "%s: wrote %d segments to %s (%s)",
                split, len(rows[split]), manifest,
                ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
            )
