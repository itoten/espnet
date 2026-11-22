"""Tests for the cls system's inference runners."""

import numpy as np
import pytest
import torch

from espnet3.contract.inference import load_inference_contract
from espnet3.systems.cls.src.inference_runner import (
    ClsInferenceRunner,
    PrefetchInferenceRunner,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_writes_hyp_and_score           | One item: the contracted     |
# |                                             | outputs carry label + scores.|
# | test_forward_handles_a_batch                | A batch reaches the model as |
# |                       | a list of arrays and gives one record per item. |
# | test_format_output_passes_legacy_records    | A record built by a legacy   |
# |                                             | output_fn is left as it is.  |
# | test_prefetch_reads_each_item_once          | The next batch is read ahead |
# |                         | and that read is reused, not repeated.        |
# | test_prefetch_drops_reads_for_skipped_batch | A read for a batch that does |
# |                         | not come next is not kept.                    |
# | test_prefetch_takes_read_workers_from_env   | The thread count comes from  |
# |                                             | the worker's env.            |
# | test_prefetch_hands_read_workers_to_provider| The runner puts the thread   |
# |                         | count in the provider's params for workers.   |
# | test_prefetch_rejects_zero_read_workers     | At least one thread.         |

CONTRACT = {
    "inputs": {"speech": {"kind": "audio", "sample_rate": 16000, "channels": 1}},
    "outputs": {"hyp": {"kind": "text"}, "score": {"kind": "text"}},
}
LABELS = ["angry", "happy"]


def _result(speech):
    """Stand-in for ClsInferenceModel: longer than 800 samples is happy."""
    index = int(len(speech) > 800)
    scores = torch.tensor([0.75, 0.25] if index == 0 else [0.25, 0.75])
    return [index], scores, LABELS[index]


def _model(speech):
    if isinstance(speech, list):
        return [_result(s) for s in speech]
    return _result(speech)


def _item(samples):
    return {"speech": np.zeros(samples, dtype=np.float32), "label": "happy"}


def test_forward_writes_hyp_and_score():
    seen = {}

    def model(speech):
        seen["speech"] = speech
        return _model(speech)

    out = ClsInferenceRunner.forward(
        0,
        dataset=[_item(1600)],
        model=model,
        input_key="speech",
        inference_contract=load_inference_contract(CONTRACT),
    )

    assert out == {"hyp": "happy", "score": "0.25 0.75", "utt_id": "0"}
    assert isinstance(seen["speech"], np.ndarray)
    assert seen["speech"].shape == (1600,)


def test_forward_handles_a_batch():
    seen = {}

    def model(speech):
        seen["speech"] = speech
        return _model(speech)

    out = ClsInferenceRunner.forward(
        [0, 1],
        dataset=[_item(400), _item(1600)],
        model=model,
        input_key="speech",
        inference_contract=load_inference_contract(CONTRACT),
    )

    assert [type(s) for s in seen["speech"]] == [np.ndarray, np.ndarray]
    assert [o["hyp"] for o in out] == ["angry", "happy"]
    assert [o["utt_id"] for o in out] == ["0", "1"]


def test_format_output_passes_legacy_records():
    record = {"utt_id": "u1", "hyp": "joy", "ref": "joy", "score": "1.0"}

    assert ClsInferenceRunner.format_output(record, batched=False) is record
    assert ClsInferenceRunner.format_output([record], batched=True) == [record]


class CountingDataset(list):
    """A list that records which items were read."""

    def __init__(self, items):
        super().__init__(items)
        self.reads = []

    def __getitem__(self, idx):
        self.reads.append(idx)
        return super().__getitem__(idx)


@pytest.fixture(autouse=True)
def _fresh_prefetch_state():
    """Each test starts without a pool or reads in flight."""
    yield
    if PrefetchInferenceRunner._pool is not None:
        PrefetchInferenceRunner._pool.shutdown(wait=True)
    PrefetchInferenceRunner._pool = None
    PrefetchInferenceRunner._pool_size = 0
    PrefetchInferenceRunner._pending = {}


def _prefetch(idx, dataset, **kwargs):
    return PrefetchInferenceRunner.forward(
        idx, dataset=dataset, model=_model, input_key="speech", **kwargs
    )


def test_prefetch_reads_each_item_once():
    dataset = CountingDataset([_item(1600) for _ in range(4)])

    first = _prefetch([0, 1], dataset)
    second = _prefetch([2, 3], dataset)
    PrefetchInferenceRunner._pool.shutdown(wait=True)

    assert [o["hyp"] for o in first + second] == ["happy"] * 4
    assert sorted(dataset.reads) == [0, 1, 2, 3]


def test_prefetch_drops_reads_for_skipped_batch():
    dataset = CountingDataset([_item(1600) for _ in range(10)])

    _prefetch([0, 1], dataset)
    _prefetch([6, 7], dataset)

    pending = {i for _, i in PrefetchInferenceRunner._pending}
    assert pending == {8, 9}


def test_prefetch_takes_read_workers_from_env():
    _prefetch([0], CountingDataset([_item(1600)]), read_workers=3)

    assert PrefetchInferenceRunner._pool_size == 3


class FakeProvider:
    def __init__(self):
        self.params = {"input_key": "speech"}


def test_prefetch_hands_read_workers_to_provider():
    provider = FakeProvider()

    PrefetchInferenceRunner(provider=provider, read_workers=4)

    assert provider.params["read_workers"] == 4


def test_prefetch_rejects_zero_read_workers():
    with pytest.raises(ValueError, match="read_workers"):
        PrefetchInferenceRunner(provider=FakeProvider(), read_workers=0)
