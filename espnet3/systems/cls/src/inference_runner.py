"""Inference runners of the cls system."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Mapping, Optional, Tuple

from espnet3.systems.base.inference_runner import InferenceRunner

__all__ = ["ClsInferenceRunner", "PrefetchInferenceRunner"]


def _format_one(result) -> Dict[str, str]:
    """Name the label and the per-class scores of one ``(index, scores, label)``."""
    _, scores, label = result
    return {"hyp": label, "score": " ".join(str(value) for value in scores.tolist())}


class ClsInferenceRunner(InferenceRunner):
    """Turn the cls inference models' results into ``hyp`` and ``score``.

    ``ClsInferenceModel`` and the zero-shot models next to it return
    ``(prediction, scores, label)`` for one waveform and a list of those for
    a batch. This runner writes the label as ``hyp`` and the space-separated
    class probabilities, in ``token_list`` order, as ``score``. The reference
    is not written: the metrics read it from the data (``dataset:label``).

    Examples:
        In a recipe's ``conf/inference.yaml``:

        .. code-block:: yaml

            input_key: speech
            contract:
              inputs:
                speech: {kind: audio, sample_rate: 16000, channels: 1}
              outputs:
                hyp: {kind: text}
                score: {kind: text}
            runner:
              _target_: espnet3.systems.cls.src.inference_runner.ClsInferenceRunner
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Pass the checked audio to the model as plain arrays.

        Args:
            inputs: The model's keyword arguments; ``speech`` holds an
                ``Audio``, or a list of them for a batch.

        Returns:
            A copy of ``inputs`` whose ``speech`` holds the samples.

        Examples:
            >>> import numpy as np
            >>> from espnet3.contract.kinds.audio import Audio
            >>> audio = Audio(np.zeros(4, dtype=np.float32), 16000)
            >>> ClsInferenceRunner.prepare_model_inputs({"speech": audio})["speech"]
            array([0., 0., 0., 0.], dtype=float32)
        """
        speech = inputs["speech"]
        if isinstance(speech, (list, tuple)):
            return {**inputs, "speech": [getattr(s, "array", s) for s in speech]}
        return {**inputs, "speech": getattr(speech, "array", speech)}

    @staticmethod
    def format_output(result, *, batched):
        """Name the predicted label and the per-class scores.

        Args:
            result: ``(prediction, scores, label)``, or a list of them when
                ``batched``.
            batched: Whether the call was a batch.

        Returns:
            ``{"hyp": <label>, "score": <space-separated scores>}``, or a list
            of them when ``batched``. A result that is already a mapping (a
            legacy config whose ``output_fn`` built the record) is returned
            as it is.

        Examples:
            >>> import torch
            >>> result = ([1], torch.tensor([0.25, 0.75]), "joy")
            >>> ClsInferenceRunner.format_output(result, batched=False)
            {'hyp': 'joy', 'score': '0.25 0.75'}
        """
        if batched:
            return [
                item if isinstance(item, Mapping) else _format_one(item)
                for item in result
            ]
        return result if isinstance(result, Mapping) else _format_one(result)


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


class PrefetchInferenceRunner(ClsInferenceRunner):
    """``ClsInferenceRunner`` that reads batches with threads, one batch ahead.

    The base runner reads the items of a batch one after another in the
    process that runs the model, so the GPU waits while audio files are read.
    With ``batch_size`` set, this runner reads the items of a batch with a
    pool of threads and starts reading the next batch (the next
    ``batch_size`` indices) before the model runs on the current one. Without
    ``batch_size`` it behaves like ``ClsInferenceRunner``.

    A read that was started for a batch that never came (the end of a shard,
    or a resumed run that skips batches) is dropped at the next batch, so
    the waveforms it holds are not kept.

    Args:
        *args: Forwarded to ``InferenceRunner``.
        read_workers: Threads that read audio files, per inference worker.
            It reaches the workers through the provider's ``params``, so it
            also applies when inference runs in parallel processes.
        **kwargs: Forwarded to ``InferenceRunner``.

    Examples:
        In a recipe's ``conf/inference.yaml``:

        .. code-block:: yaml

            batch_size: 256
            runner:
              _target_: espnet3.systems.cls.src.inference_runner.PrefetchInferenceRunner
              read_workers: 16
    """

    DEFAULT_READ_WORKERS = 16

    # forward() is a classmethod that runs in each inference worker process,
    # so the pool and the reads in flight live on the class, per process.
    _pool: Optional[ThreadPoolExecutor] = None
    _pool_size: int = 0
    _pending: Dict[Tuple[int, int], Future] = {}

    def __init__(
        self, *args, read_workers: int = DEFAULT_READ_WORKERS, **kwargs
    ) -> None:
        """Build the base runner and hand ``read_workers`` to the workers."""
        super().__init__(*args, **kwargs)
        read_workers = int(read_workers)
        if read_workers < 1:
            raise ValueError(f"read_workers must be at least 1, got {read_workers}")
        params = getattr(self.provider, "params", None)
        if isinstance(params, dict):
            params["read_workers"] = read_workers

    @classmethod
    def _get_pool(cls, read_workers: int) -> ThreadPoolExecutor:
        if cls._pool is None or cls._pool_size != read_workers:
            if cls._pool is not None:
                cls._pool.shutdown(wait=False, cancel_futures=True)
            cls._pool = ThreadPoolExecutor(max_workers=read_workers)
            cls._pool_size = read_workers
            cls._pending = {}
        return cls._pool

    @classmethod
    def forward(cls, idx, dataset=None, model=None, **kwargs):
        """Read the batch (and the next one) with threads, then run the model."""
        read_workers = int(kwargs.pop("read_workers", cls.DEFAULT_READ_WORKERS))
        if not isinstance(idx, (list, tuple)) or not idx or dataset is None:
            return super().forward(idx, dataset=dataset, model=model, **kwargs)

        pool = cls._get_pool(read_workers)
        dataset_id = id(dataset)

        def submit(i: int) -> Future:
            future = cls._pending.pop((dataset_id, i), None)
            return future if future is not None else pool.submit(dataset.__getitem__, i)

        futures = {i: submit(i) for i in idx}

        try:
            size = len(dataset)
        except TypeError:
            size = None
        start = int(idx[-1]) + 1
        stop = start + len(idx) if size is None else min(start + len(idx), size)
        upcoming = {(dataset_id, i) for i in range(start, stop)}
        # Reads for any other batch were started for a batch that did not
        # come next; drop them so they do not hold their audio.
        for key in list(cls._pending):
            if key not in upcoming:
                cls._pending.pop(key).cancel()
        for key in upcoming:
            if key not in cls._pending:
                cls._pending[key] = pool.submit(dataset.__getitem__, key[1])

        items = {i: futures[i].result() for i in idx}
        return super().forward(idx, dataset=_Prefetched(items), model=model, **kwargs)
