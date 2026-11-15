"""Inference runner that reads the audio of a batch ahead of the model."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Mapping, Optional, Tuple

from espnet3.systems.base.inference_runner import InferenceRunner

__all__ = ["PrefetchInferenceRunner"]


class _Prefetched(Mapping):
    """The items of one batch, already read, indexed like the dataset."""

    def __init__(self, items: Dict[int, Any]) -> None:
        self._items = items

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class PrefetchInferenceRunner(InferenceRunner):
    """``InferenceRunner`` that reads batches with threads, one batch ahead.

    The base runner reads the items of a batch one after another in the
    process that runs the model, so the GPU waits while audio files are read.
    With ``batch_size`` set, this runner reads the items of a batch with a
    pool of threads and starts reading the next batch (the next
    ``batch_size`` indices) before the model runs on the current one. Without
    ``batch_size`` it behaves like ``InferenceRunner``.

    Examples:
        In a recipe's ``conf/inference.yaml``:

        .. code-block:: yaml

            batch_size: 256
            runner:
              _target_: espnet3.systems.cls.src.inference_runner.PrefetchInferenceRunner
              read_workers: 16
    """

    read_workers = 16
    _pool: Optional[ThreadPoolExecutor] = None
    _pending: Dict[Tuple[int, int], Future] = {}

    def __init__(self, *args, read_workers: int = 16, **kwargs) -> None:
        """Set the number of reading threads, then build the base runner.

        Args:
            *args: Forwarded to ``InferenceRunner``.
            read_workers: Threads that read audio files.
            **kwargs: Forwarded to ``InferenceRunner``.
        """
        super().__init__(*args, **kwargs)
        # forward() is a classmethod, so the pool lives on the class.
        type(self).read_workers = int(read_workers)

    @classmethod
    def _submit(cls, dataset, idx: int) -> Future:
        if cls._pool is None:
            cls._pool = ThreadPoolExecutor(max_workers=cls.read_workers)
        key = (id(dataset), idx)
        future = cls._pending.pop(key, None)
        if future is None:
            future = cls._pool.submit(dataset.__getitem__, idx)
        return future

    @classmethod
    def forward(cls, idx, dataset=None, model=None, **kwargs):
        """Read the batch (and the next one) with threads, then run the model."""
        if not isinstance(idx, (list, tuple)) or not idx or dataset is None:
            return super().forward(idx, dataset=dataset, model=model, **kwargs)
        futures = {i: cls._submit(dataset, i) for i in idx}
        try:
            size = len(dataset)
        except TypeError:
            size = None
        start = int(idx[-1]) + 1
        stop = start + len(idx) if size is None else min(start + len(idx), size)
        for i in range(start, stop):
            key = (id(dataset), i)
            if key not in cls._pending:
                cls._pending[key] = cls._submit(dataset, i)
        items = {i: futures[i].result() for i in idx}
        return super().forward(
            idx, dataset=_Prefetched(items), model=model, **kwargs
        )
