import pytest

from egs3.multidataset_ser_03.cls.dataset import builder


def _rows():
    return {
        "a": {"train": [("a-1", "/a1.wav", "happy")], "test": [("a-2", "/a2.wav", "sad")]},
        "b": {"train": [("b-1", "/b1.wav", "angry")], "valid": [("b-2", "/b2.wav", "fear")]},
        "c": {"train": [("c-1", "/c1.wav", "neutral")]},
    }


def test_held_out_corpora_takes_whole_corpora():
    train, valid = builder.held_out_corpora(_rows(), {"corpora": ["b"]})
    assert sorted(valid) == [("b-1", "/b1.wav", "angry"), ("b-2", "/b2.wav", "fear")]
    assert sorted(u for u, _, _ in train) == ["a-1", "a-2", "c-1"]


@pytest.mark.parametrize(
    "corpora, message",
    [([], "at least one"), (["b", "b"], "repeats"), (["x"], "not training corpora")],
)
def test_held_out_corpora_rejects_bad_lists(corpora, message):
    with pytest.raises(ValueError, match=message):
        builder.held_out_corpora(_rows(), {"corpora": corpora})


def test_recipe_holds_out_the_five_dev_corpora_from_training():
    validation = builder._CFG["validation"]
    assert validation["method"] == "held_out_corpora"
    held = list(validation["held_out_corpora"]["corpora"])
    assert held == ["berst", "csemotions", "eseravd", "thorsten", "uftbesd"]
    assert set(held) <= set(builder.TRAIN_CORPORA)
    assert not set(held) & set(builder.EVAL_CORPORA)
