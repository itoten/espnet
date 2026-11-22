"""Tests that every cls recipe's inference and metrics configs use the contract."""

from pathlib import Path

import pytest
from hydra.utils import get_class

from espnet3.systems.cls.src.inference_runner import ClsInferenceRunner
from espnet3.utils.config_utils import load_and_merge_config

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_inference_config_uses_the_contract     | Merged with the template,    |
# |                          | every inference config declares hyp and score, |
# |                          | runs a ClsInferenceRunner and has no output_fn |
# |                          | (infer refuses output_fn with a contract).     |
# | test_metrics_read_the_reference_from_data   | Every metric binds ref to    |
# |                                             | dataset:label.               |

EGS3 = Path(__file__).resolve().parents[4] / "egs3"
DEFAULTS = "egs3.TEMPLATE.cls"


def _configs(prefix):
    paths = sorted(EGS3.glob(f"*/cls/conf/{prefix}*.yaml"))
    return pytest.mark.parametrize(
        "path", paths, ids=[str(p.relative_to(EGS3)) for p in paths]
    )


@_configs("inference")
def test_inference_config_uses_the_contract(path):
    config = load_and_merge_config(
        path, config_name="inference.yaml", default_package=DEFAULTS, resolve=False
    )

    assert not config.get("output_fn")
    assert config.input_key == "speech"
    assert set(config.contract.outputs) == {"hyp", "score"}
    assert config.contract.inputs.speech.sample_rate == 16000
    assert issubclass(get_class(config.runner._target_), ClsInferenceRunner)


@_configs("metrics")
def test_metrics_read_the_reference_from_data(path):
    config = load_and_merge_config(
        path, config_name="metrics.yaml", default_package=DEFAULTS, resolve=False
    )

    for metric in config.metrics:
        assert metric.inputs.ref == "dataset:label"
        assert metric.metric._target_.startswith("espnet3.systems.esp2_cls.metrics.")
