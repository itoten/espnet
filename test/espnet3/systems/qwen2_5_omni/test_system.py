"""Tests for the Qwen2.5-Omni system."""

import pytest
from omegaconf import OmegaConf

import espnet3.systems.base.system as base_system
from espnet3.systems.qwen2_5_omni.system import Qwen2_5OmniSystem, _parse_only

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                  | Description                   |
# |--------------------------------------------|-------------------------------|
# | test_declares_only_infer_and_measure       | No training stages.           |
# | test_parse_only                            | None / "" / string / list.    |
# | test_infer_runs_every_test_set_by_default  | `only` unset passes the       |
# |                                            | config through.               |
# | test_infer_keeps_only_named_test_sets      | `only` filters dataset.test   |
# |                                            | and restores the config.      |
# | test_infer_rejects_unknown_names           | Typos fail before inference.  |


def _config(only=None):
    return OmegaConf.create(
        {
            "exp_tag": "t",
            "inference_dir": "exp/t/inference",
            "only": only,
            "dataset": {"test": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        }
    )


@pytest.fixture
def seen(monkeypatch):
    calls = []
    monkeypatch.setattr(
        base_system,
        "infer",
        lambda config: calls.append([e.name for e in config.dataset.test]),
    )
    return calls


def test_declares_only_infer_and_measure():
    assert [s.name for s in Qwen2_5OmniSystem.stages] == ["infer", "measure"]


def test_parse_only():
    assert _parse_only(None) is None
    assert _parse_only("") is None
    assert _parse_only("a  c") == ["a", "c"]
    assert _parse_only(["b"]) == ["b"]


def test_infer_runs_every_test_set_by_default(seen):
    Qwen2_5OmniSystem(inference_config=_config()).infer()

    assert seen == [["a", "b", "c"]]


def test_infer_keeps_only_named_test_sets(seen):
    config = _config(only="c a")
    system = Qwen2_5OmniSystem(inference_config=config)

    system.infer()

    assert seen == [["a", "c"]]
    assert [e.name for e in system.inference_config.dataset.test] == ["a", "b", "c"]
    assert [e.name for e in config.dataset.test] == ["a", "b", "c"]


def test_infer_rejects_unknown_names(seen):
    system = Qwen2_5OmniSystem(inference_config=_config(only=["a", "x"]))

    with pytest.raises(ValueError, match="x"):
        system.infer()
    assert seen == []
