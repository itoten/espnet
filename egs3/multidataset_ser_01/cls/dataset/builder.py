"""Prepare every corpus of experiment multidataset_ser_01 through its own recipe.

Each corpus is downloaded and converted by its own ``egs3/<corpus>/cls``
``DatasetBuilder``, as ``egs3/multicorpus_ood20/cls`` does. This builder adds
what preparing 72 corpora in one run needs: a corpus that fails does not stop
the others, network errors that can pass are retried with a growing wait, and
the run ends with a table of what is ready and why the rest failed.

The merged training and evaluation manifests are not written yet: which corpus
is used for training and which for evaluation is still being decided
(``docs/discussions/0003-license-classes.md`` and ``0004-label-mapping.md`` of
ser_dataset_db).
"""

from __future__ import annotations

import http.client
import logging
import socket
import ssl
import subprocess
import time
import traceback
import urllib.error
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Sequence

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.components.data.dataset_module import load_dataset_module
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
CORPORA: list[str] = [str(name) for name in _CFG["corpora"]]

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


class MultidatasetSer01Builder(DatasetBuilder):
    """Prepare the 72 corpora of experiment multidataset_ser_01.

    ``prepare_source`` runs each corpus's own builder, one corpus at a time.
    ``build`` has nothing to write yet (see the module docstring).
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
        """Report built: no merged manifest is defined yet."""
        return True

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """No-op until the training and evaluation corpora are decided."""
        return None
