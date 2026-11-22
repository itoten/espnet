"""Tests for the cls system stage hooks."""

from collections import Counter
from pathlib import Path

import pytest

from espnet3.systems.cls.system import ClsSystem

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_class_weights_are_inverse_frequency    | w_c = N / (C * n_c), the     |
# |                          | definition sklearn calls "balanced".            |
# | test_class_weights_skip_added_symbols       | <unk> carries no count, so   |
# |                          | the file lines up with the predicted classes.   |
# | test_class_weights_follow_token_list_order  | The order matches the token  |
# |                          | list, not the counter's insertion order.        |
# | test_balanced_classes_all_weigh_one         | Equal counts leave every     |
# |                                             | weight at 1.
# | test_class_weights_reject_a_symbol_inserted_first | A class with no  |
# |                          | example has no weight, so it is an error.       |
# | test_prepare_labels_writes_class_weights    | The stage writes the label   |
# |                          | list and one weight per class next to it.       |
# | test_prepare_labels_without_weights_filename | Only the label list, as in  |
# |                                             | esp2_cls.                    |
# | test_stages_are_those_of_esp2_cls           | The data stages are shared.  |


def _weights(tmp_path: Path, counter: Counter, labels: list[str]) -> list[float]:
    output = tmp_path / "class_weights"
    ClsSystem._write_class_weights(counter, labels, output)
    return [float(line) for line in output.read_text().splitlines()]


def test_class_weights_are_inverse_frequency(tmp_path: Path):
    counter = Counter({"neutral": 60, "anger": 30, "fear": 10})
    labels = ["neutral", "anger", "fear", "<unk>"]

    weights = _weights(tmp_path, counter, labels)

    total, classes = 100, 3
    expected = [total / (classes * n) for n in (60, 30, 10)]
    assert weights == pytest.approx(expected, rel=1e-9)


def test_class_weights_skip_added_symbols(tmp_path: Path):
    counter = Counter({"neutral": 2, "anger": 2})
    labels = ["neutral", "anger", "<unk>"]

    assert len(_weights(tmp_path, counter, labels)) == 2


def test_class_weights_follow_token_list_order(tmp_path: Path):
    """The rarest class weighs most, wherever the counter happened to put it."""
    counter = Counter({"anger": 10, "neutral": 90})
    labels = ["neutral", "anger", "<unk>"]

    weights = _weights(tmp_path, counter, labels)

    assert weights[0] < weights[1]


def test_balanced_classes_all_weigh_one(tmp_path: Path):
    counter = Counter({"a": 5, "b": 5, "c": 5})
    labels = ["a", "b", "c", "<unk>"]

    assert _weights(tmp_path, counter, labels) == pytest.approx([1.0, 1.0, 1.0])


def _collect_stats_with(monkeypatch, model_cfg):
    from omegaconf import OmegaConf

    seen = {}
    monkeypatch.setattr(
        "espnet3.systems.base.system.collect_stats",
        lambda cfg: seen.setdefault("model", OmegaConf.to_container(cfg.model)),
    )
    cfg = OmegaConf.create({"exp_dir": "exp", "stats_dir": "stats", "model": model_cfg})
    system = ClsSystem(training_config=cfg)
    system.collect_stats()
    return cfg, seen["model"]


def test_collect_stats_drops_frontend_normalize(monkeypatch):
    """The normalizer needs the stats this stage creates, so it is removed."""
    model = {"frontend": {"_target_": "x", "normalize": {"stats_file": "a.npz"}}}

    cfg, seen = _collect_stats_with(monkeypatch, model)

    assert "normalize" not in seen["frontend"]
    assert "normalize" in cfg.model.frontend


def test_collect_stats_without_normalize_is_untouched(monkeypatch):
    for model in ({"frontend": {"_target_": "x"}}, {"_target_": "y"}):
        _, seen = _collect_stats_with(monkeypatch, model)
        assert seen == model


def test_class_weights_reject_a_symbol_inserted_first(tmp_path: Path):
    counter = Counter({"neutral": 2, "anger": 2})
    labels = ["<blank>", "neutral", "anger", "<unk>"]

    with pytest.raises(ValueError, match="<blank>"):
        _weights(tmp_path, counter, labels)


def _prepare_labels_config(tmp_path: Path, **extra):
    from omegaconf import OmegaConf

    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "u1\ta.wav\tneutral\ts1\n"
        "u2\tb.wav\tneutral\ts1\n"
        "u3\tc.wav\tneutral\ts2\n"
        "u4\td.wav\tanger\ts2\n"
    )
    return OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "exp"),
            "prepare_labels": {
                "save_path": str(tmp_path / "data"),
                "filename": "token_list",
                "manifest_path": str(manifest),
                "add_symbol": ["<unk>:-1"],
                **extra,
            },
        }
    )


def test_prepare_labels_writes_class_weights(tmp_path: Path):
    config = _prepare_labels_config(tmp_path, weights_filename="class_weights")

    ClsSystem(training_config=config).prepare_labels()

    data = tmp_path / "data"
    assert (data / "token_list").read_text().split() == ["neutral", "anger", "<unk>"]
    weights = [float(w) for w in (data / "class_weights").read_text().split()]
    assert weights == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])


def test_prepare_labels_without_weights_filename(tmp_path: Path):
    config = _prepare_labels_config(tmp_path)

    ClsSystem(training_config=config).prepare_labels()

    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["token_list"]


def test_stages_are_those_of_esp2_cls():
    from espnet3.systems.esp2_cls.system import ESP2CLSSystem

    assert ClsSystem.stages == ESP2CLSSystem.stages
