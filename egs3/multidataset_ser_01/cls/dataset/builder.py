"""Prepare the corpora of experiment multidataset_ser_01 and merge their manifests.

``prepare_source`` downloads and converts each corpus with its own
``egs3/<corpus>/cls`` ``DatasetBuilder``, as ``egs3/multicorpus_ood20/cls``
does, and adds what preparing 72 corpora in one run needs: a corpus that fails
does not stop the others, network errors that can pass are retried with a
growing wait, and the run ends with a table of what is ready and why the rest
failed.

``build`` then merges the corpora's own manifests (every split, unfiltered)
into this recipe's: one training and one validation manifest from the
training corpora, keeping only the labels in ``train_labels``, and one
manifest per evaluation corpus with every row. How the validation rows are
taken out of the training corpora is chosen by ``validation.method`` in
``dataset/config.yaml``; see :data:`VALIDATION_METHODS`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import random
import socket
import ssl
import subprocess
import time
import traceback
import urllib.error
from collections import Counter
from dataclasses import dataclass
from importlib import import_module, resources
from pathlib import Path
from typing import Callable, Mapping, Sequence

from omegaconf import OmegaConf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.components.data.dataset_module import load_dataset_module
from espnet3.components.data.multicorpus_manifest import write_manifest
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
CORPORA: list[str] = [str(name) for name in _CFG["corpora"]]
TRAIN_CORPORA: list[str] = [str(name) for name in _CFG["train_corpora"]]
EVAL_CORPORA: list[str] = [str(name) for name in _CFG["eval_corpora"]]
TRAIN_LABELS: list[str] = [str(label) for label in _CFG["train_labels"]]
_LABEL_MAP: dict[str, str] = {str(k): str(v) for k, v in _CFG["label_map"].items()}
# The splits every corpus recipe writes to its own `manifest/`.
CORPUS_SPLITS = ("train", "valid", "test")

Row = tuple[str, str, str]  # (utt_id, wav_path, label)

# Exception class names, from libraries some corpus recipes use (requests,
# urllib3, huggingface_hub), that mean a connection broke or timed out.
_TRANSIENT_NAMES = {
    "ChunkedEncodingError",
    "ConnectTimeout",
    "ConnectTimeoutError",
    "ConnectionError",
    "IncompleteRead",
    "ProtocolError",
    "ReadTimeout",
    "ReadTimeoutError",
    "RemoteDisconnected",
    "Timeout",
}


@dataclass
class CorpusResult:
    """Outcome of preparing one corpus."""

    name: str
    ok: bool
    attempts: int
    seconds: float
    error: str = ""


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return ``exc`` and the exceptions it was raised from, outermost first."""
    chain: list[BaseException] = []
    while exc is not None and exc not in chain and len(chain) < 10:
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    return chain


def is_transient(exc: BaseException) -> bool:
    """Tell whether waiting and trying again can make ``exc`` go away.

    Args:
        exc: Exception raised while preparing a corpus.

    Returns:
        bool: True for broken or timed-out connections, HTTP 5xx, 408 and 429,
        and a failed ``git``/``kaggle`` call. False for HTTP 4xx, a server
        certificate that cannot be trusted, and everything else (missing files
        or tokens, bad archives, wrong checksums).

    Examples:
        >>> is_transient(TimeoutError())
        True
        >>> is_transient(FileNotFoundError("corpus missing"))
        False
    """
    for err in _exception_chain(exc):
        if isinstance(err, ssl.SSLCertVerificationError):
            return False
        if isinstance(err, urllib.error.HTTPError):
            return err.code >= 500 or err.code in (408, 429)
        if isinstance(err, urllib.error.URLError):
            if isinstance(err.reason, ssl.SSLCertVerificationError):
                return False
            return True
        if isinstance(
            err,
            (
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
                ssl.SSLError,
                socket.gaierror,
                subprocess.CalledProcessError,
            ),
        ):
            return True
        if type(err).__name__ in _TRANSIENT_NAMES:
            return True
    return False


def _summarize(exc: BaseException) -> str:
    """Return one line naming the error and its first message line."""
    message = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {message[0] if message else ''}"


def prepare_corpus(name: str, attempts: int, wait_seconds: float) -> CorpusResult:
    """Download and convert one corpus with its own recipe, retrying if useful.

    Args:
        name: Corpus recipe directory under ``egs3/``, e.g. ``ravdess``.
        attempts: Tries in all before giving up on a transient error.
        wait_seconds: Wait before the first retry; doubled after each failure.

    Returns:
        CorpusResult: Whether the corpus is ready, and the error if not.
    """
    start = time.perf_counter()
    try:
        module = load_dataset_module(data_src=f"{name}/cls")
    except Exception as exc:  # noqa: BLE001 - reported per corpus
        logger.error("%s: cannot load its recipe\n%s", name, traceback.format_exc())
        return CorpusResult(name, False, 0, time.perf_counter() - start, _summarize(exc))

    recipe_root = Path(module.__file__).resolve().parents[1]
    builder = module.DatasetBuilder()
    wait = float(wait_seconds)
    for attempt in range(1, attempts + 1):
        try:
            if not builder.is_source_prepared(recipe_dir=recipe_root):
                logger.info("%s: downloading (attempt %d/%d)", name, attempt, attempts)
                builder.prepare_source(recipe_dir=recipe_root)
            if not builder.is_built(recipe_dir=recipe_root):
                logger.info("%s: building", name)
                builder.build(recipe_dir=recipe_root)
            return CorpusResult(name, True, attempt, time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 - one corpus must not stop the rest
            if attempt < attempts and is_transient(exc):
                logger.warning(
                    "%s: %s; retrying in %.0f s", name, _summarize(exc), wait
                )
                time.sleep(wait)
                wait *= 2
                continue
            logger.error("%s: failed\n%s", name, traceback.format_exc())
            return CorpusResult(
                name, False, attempt, time.perf_counter() - start, _summarize(exc)
            )
    raise AssertionError("unreachable")


def _selected(only: str | Sequence[str] | None) -> list[str]:
    """Return the corpora to prepare: all of them, or those named in ``only``.

    Raises:
        ValueError: If ``only`` names a corpus that is not in the list.
    """
    if not only:
        return list(CORPORA)
    names = only.split() if isinstance(only, str) else [str(n) for n in only]
    unknown = [n for n in names if n not in CORPORA]
    if unknown:
        raise ValueError(f"Not corpora of this experiment: {', '.join(unknown)}")
    return names


def _is_ready(name: str) -> bool:
    """Tell whether a corpus is already downloaded and converted."""
    try:
        module = load_dataset_module(data_src=f"{name}/cls")
        recipe_root = Path(module.__file__).resolve().parents[1]
        builder = module.DatasetBuilder()
        return builder.is_source_prepared(recipe_dir=recipe_root) and builder.is_built(
            recipe_dir=recipe_root
        )
    except Exception:  # noqa: BLE001 - anything wrong means "not ready"
        return False


def resolve_data_root(recipe_root: Path) -> Path:
    """Return the directory this recipe's merged manifests are written into.

    Args:
        recipe_root: Recipe directory, used to resolve relative locations.

    Returns:
        Path: ``$MULTIDATASET_SER_01_OUTPUT`` when set, else
        ``<recipe_dir>/data``.
    """
    env_value = os.environ.get(str(_CFG["output_env_var"]))
    if env_value:
        return Path(env_value).expanduser()
    return recipe_root / str(_CFG["data_path"])


def check_roles() -> None:
    """Check that every corpus of ``corpora`` has exactly one role.

    Raises:
        ValueError: If a corpus has no role or two, or a role names a corpus
            that is not in ``corpora``.
    """
    train, evals, corpora = set(TRAIN_CORPORA), set(EVAL_CORPORA), set(CORPORA)
    problems = []
    if train & evals:
        problems.append(f"in both roles: {sorted(train & evals)}")
    if corpora - train - evals:
        problems.append(f"without a role: {sorted(corpora - train - evals)}")
    if (train | evals) - corpora:
        problems.append(f"not in `corpora`: {sorted((train | evals) - corpora)}")
    if problems:
        raise ValueError("dataset/config.yaml roles: " + "; ".join(problems))


def eval_manifest_path(name: str) -> str:
    """Return the manifest path, under the data root, of one evaluation corpus."""
    return f"manifest/{name}.tsv"


def read_corpus_rows(name: str) -> dict[str, list[Row]]:
    """Read every split of one corpus recipe's unfiltered manifest.

    The corpus must already be prepared (``prepare_source`` does that).

    Args:
        name: Corpus recipe directory under ``egs3/``, e.g. ``ravdess``.

    Returns:
        dict[str, list[Row]]: Rows of each split in :data:`CORPUS_SPLITS`.

    Raises:
        FileNotFoundError: If a manifest of the corpus is missing.
    """
    module = load_dataset_module(data_src=f"{name}/cls")
    recipe_root = Path(module.__file__).resolve().parents[1]
    data_root = import_module(f"{module.__name__}.builder").resolve_data_root(
        recipe_root
    )
    corpus_cfg = load_config_with_defaults(
        str(recipe_root / "dataset" / "config.yaml"), resolve=False
    )["builder"]
    rows: dict[str, list[Row]] = {}
    for split in CORPUS_SPLITS:
        path = data_root / str(corpus_cfg["manifest_paths"][split])
        with path.open(encoding="utf-8") as fh:
            rows[split] = [
                tuple(line.rstrip("\n").split("\t", maxsplit=2))
                for line in fh
                if line.strip()
            ]
    return rows


def map_label(label: str) -> str:
    """Apply ``label_map`` (spelling fixes) to one label."""
    return _LABEL_MAP.get(label, label)


# --------------------------------------------------------------------------
# Validation selection
# --------------------------------------------------------------------------
# A method takes the training corpora's rows, keyed by corpus and then by the
# corpus recipe's own split, already label-mapped and label-filtered, and the
# method's own block of `validation` in `dataset/config.yaml`. It returns the
# training rows and the validation rows. To add a method, write a function
# with this signature, register it in VALIDATION_METHODS and give it a block
# in the config.
ValidationMethod = Callable[
    [Mapping[str, Mapping[str, list[Row]]], Mapping],
    tuple[list[Row], list[Row]],
]


def _corpus_seed(seed: int, corpus: str) -> int:
    """Derive a per-corpus seed that does not depend on the other corpora."""
    digest = hashlib.sha256(f"{seed}:{corpus}".encode()).hexdigest()
    return int(digest[:16], 16)


def random_per_corpus(
    rows: Mapping[str, Mapping[str, list[Row]]], params: Mapping
) -> tuple[list[Row], list[Row]]:
    """Take ``ratio`` of each corpus's rows at random as validation.

    The corpus's own splits are pooled and sorted by utterance id first, so
    the choice depends only on ``seed``, the corpus name and its rows.
    Speakers are not taken into account. At least one row is taken from
    every corpus.

    Args:
        rows: Rows per corpus and per corpus split.
        params: ``ratio`` (0 < ratio < 1) and ``seed``.

    Returns:
        tuple[list[Row], list[Row]]: Training rows and validation rows.

    Raises:
        ValueError: If ``ratio`` is not in (0, 1).
    """
    ratio = float(params["ratio"])
    if not 0 < ratio < 1:
        raise ValueError(f"validation.random_per_corpus.ratio must be in (0, 1): {ratio}")
    seed = int(params["seed"])
    train: list[Row] = []
    valid: list[Row] = []
    for corpus, splits in rows.items():
        pooled = sorted(row for split_rows in splits.values() for row in split_rows)
        if not pooled:
            continue
        n_valid = max(1, round(ratio * len(pooled)))
        chosen = set(random.Random(_corpus_seed(seed, corpus)).sample(range(len(pooled)), n_valid))
        for index, row in enumerate(pooled):
            (valid if index in chosen else train).append(row)
        logger.info(
            "validation: %s: %d of %d rows", corpus, n_valid, len(pooled)
        )
    return train, valid


def corpus_valid_split(
    rows: Mapping[str, Mapping[str, list[Row]]], params: Mapping
) -> tuple[list[Row], list[Row]]:
    """Use each corpus recipe's own ``valid`` split as validation.

    Args:
        rows: Rows per corpus and per corpus split.
        params: Unused (the method has no settings).

    Returns:
        tuple[list[Row], list[Row]]: The ``train`` and ``test`` rows of every
        corpus, and the ``valid`` rows.
    """
    train: list[Row] = []
    valid: list[Row] = []
    for corpus, splits in rows.items():
        for split, split_rows in splits.items():
            (valid if split == "valid" else train).extend(split_rows)
        logger.info(
            "validation: %s: %d of %d rows",
            corpus,
            len(splits.get("valid", [])),
            sum(len(r) for r in splits.values()),
        )
    return train, valid


VALIDATION_METHODS: dict[str, ValidationMethod] = {
    "random_per_corpus": random_per_corpus,
    "corpus_valid_split": corpus_valid_split,
}


def build_settings() -> dict:
    """Return the settings the merged manifests depend on.

    They are written next to the manifests; when they change, ``is_built``
    reports the manifests as stale so that ``create_dataset`` rewrites them.

    Raises:
        ValueError: If ``validation.method`` is not a known method.
    """
    validation = _CFG["validation"]
    method = str(validation["method"])
    if method not in VALIDATION_METHODS:
        raise ValueError(
            f"Unknown validation.method {method!r}; known: {sorted(VALIDATION_METHODS)}"
        )
    params = validation.get(method) or {}
    return {
        "train_corpora": TRAIN_CORPORA,
        "eval_corpora": EVAL_CORPORA,
        "train_labels": TRAIN_LABELS,
        "label_map": _LABEL_MAP,
        "validation": {
            "method": method,
            "params": OmegaConf.to_container(params)
            if OmegaConf.is_config(params)
            else dict(params),
        },
    }


_SETTINGS_FILE = "manifest/build_settings.json"


def _check_unique_and_present(rows: Sequence[Row], context: str) -> None:
    """Raise if an utterance id repeats or a WAV file is missing."""
    seen: set[str] = set()
    duplicates = []
    missing = []
    for utt_id, wav_path, _ in rows:
        if utt_id in seen:
            duplicates.append(utt_id)
        seen.add(utt_id)
        if not Path(wav_path).is_file():
            missing.append(wav_path)
    if duplicates:
        raise RuntimeError(f"{context}: duplicate utterance ids, e.g. {duplicates[:3]}")
    if missing:
        raise FileNotFoundError(
            f"{context}: {len(missing)} WAV file(s) do not exist, e.g. {missing[:3]}"
        )


def _lines(rows: Sequence[Row]) -> list[str]:
    return [f"{u}\t{w}\t{l}\n" for u, w, l in sorted(rows)]


def build_manifests(data_root: Path) -> None:
    """Write the training, validation and evaluation manifests.

    Args:
        data_root: Directory returned by :func:`resolve_data_root`.

    Raises:
        ValueError: If the roles or the validation method are invalid.
        RuntimeError: If an utterance id appears twice in one manifest.
        FileNotFoundError: If a listed WAV file does not exist.
    """
    check_roles()
    settings = build_settings()
    keep = set(TRAIN_LABELS)

    train_rows: dict[str, dict[str, list[Row]]] = {}
    dropped: Counter = Counter()
    for name in TRAIN_CORPORA:
        corpus_rows = read_corpus_rows(name)
        train_rows[name] = {}
        for split, rows in corpus_rows.items():
            kept = []
            for utt_id, wav_path, label in rows:
                label = map_label(label)
                if label in keep:
                    kept.append((utt_id, wav_path, label))
                else:
                    dropped[(name, label)] += 1
            train_rows[name][split] = kept
    for (name, label), n in sorted(dropped.items()):
        logger.info("train: %s: left out %d '%s' (not in train_labels)", name, n, label)

    method = settings["validation"]["method"]
    train, valid = VALIDATION_METHODS[method](
        train_rows, settings["validation"]["params"]
    )
    _check_unique_and_present(train + valid, "train+valid")
    for split, rows in (("train", train), ("valid", valid)):
        counts = Counter(label for _, _, label in rows)
        logger.info("%s: %d rows, %s", split, len(rows), dict(sorted(counts.items())))
        write_manifest(_lines(rows), data_root / str(_CFG["manifest_paths"][split]))

    for name in EVAL_CORPORA:
        rows = [
            (u, w, map_label(l))
            for split_rows in read_corpus_rows(name).values()
            for u, w, l in split_rows
        ]
        _check_unique_and_present(rows, name)
        write_manifest(_lines(rows), data_root / eval_manifest_path(name))

    settings_path = data_root / _SETTINGS_FILE
    settings_path.write_text(json.dumps(settings, indent=1, sort_keys=True) + "\n")


def manifests_built(data_root: Path) -> bool:
    """Tell whether every manifest exists and was built with the current settings."""
    paths = [str(p) for p in _CFG["manifest_paths"].values()]
    paths += [eval_manifest_path(name) for name in EVAL_CORPORA]
    if not all((data_root / p).is_file() for p in paths):
        return False
    settings_path = data_root / _SETTINGS_FILE
    if not settings_path.is_file():
        return False
    return json.loads(settings_path.read_text()) == json.loads(
        json.dumps(build_settings(), sort_keys=True)
    )


class MultidatasetSer01Builder(DatasetBuilder):
    """Prepare the corpora of experiment multidataset_ser_01 and merge them.

    ``prepare_source`` runs each corpus's own builder, one corpus at a time.
    ``build`` writes this recipe's merged manifests (see the module
    docstring).
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        only: str | Sequence[str] | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether every selected corpus is downloaded and converted."""
        return all(_is_ready(name) for name in _selected(only))

    def prepare_source(
        self,
        recipe_dir: str | Path,
        only: str | Sequence[str] | None = None,
        **_kwargs,
    ) -> None:
        """Download and convert every selected corpus with its own recipe.

        Args:
            recipe_dir: This recipe's directory (unused; each corpus resolves its
                own locations from its environment variables).
            only: Space-separated corpus names (or a list) to restrict the run
                to; all corpora when empty.

        Raises:
            RuntimeError: After every corpus has been tried, if any failed. The
                message lists each failed corpus with its error.
        """
        names = _selected(only)
        retry = _CFG["retry"]
        attempts = int(retry["attempts"])
        wait_seconds = float(retry["wait_seconds"])
        logger.info("Preparing %d corpora, one at a time", len(names))

        results = []
        for number, name in enumerate(names, start=1):
            logger.info("[%d/%d] %s", number, len(names), name)
            result = prepare_corpus(name, attempts, wait_seconds)
            logger.info(
                "[%d/%d] %s: %s (%.0f s)",
                number,
                len(names),
                name,
                "ready" if result.ok else "FAILED",
                result.seconds,
            )
            results.append(result)

        failed = [r for r in results if not r.ok]
        lines = [f"{len(results) - len(failed)} of {len(results)} corpora ready"]
        lines += [f"  FAILED {r.name}: {r.error}" for r in failed]
        logger.info("Summary:\n%s", "\n".join(lines))
        if failed:
            raise RuntimeError(
                f"{len(failed)} corpora could not be prepared (the others are "
                "ready and are skipped when create_dataset runs again):\n"
                + "\n".join(lines[1:])
            )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether the merged manifests exist and match the settings."""
        return manifests_built(resolve_data_root(Path(recipe_dir).resolve()))

    def build(
        self,
        recipe_dir: str | Path,
        only: str | Sequence[str] | None = None,
        **_kwargs,
    ) -> None:
        """Write the merged manifests, unless only some corpora were prepared.

        Args:
            recipe_dir: This recipe's directory.
            only: When set, ``prepare_source`` handled only these corpora, so
                the merged manifests, which need all of them, are not written.
        """
        if only:
            logger.info(
                "Not writing the merged manifests: MULTIDATASET_SER_01_ONLY is set"
            )
            return
        build_manifests(resolve_data_root(Path(recipe_dir).resolve()))
