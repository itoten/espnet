# test_multicorpus_manifest.py
from dataclasses import dataclass
from pathlib import Path

import pytest

from espnet3.components.data import multicorpus_manifest as mm

# ===============================================================
# Test Case Summary
# ===============================================================
# | Test Function Name                        | Description                |
# |--------------------------------------------|-----------------------------|
# | test_merge_applies_label_map_and_drops     | Unmapped labels are dropped |
# | test_merge_sorts_and_logs_per_corpus       | Output sorted by utt_id     |
# | test_merge_duplicate_utt_id_raises          | Cross-corpus id collision   |
# | test_build_eval_manifest_single_corpus     | Per-corpus eval manifest    |
# | test_write_manifest_atomic                 | write_manifest replaces file|


@dataclass(frozen=True)
class _FakeEntry:
    utt_id: str
    wav_path: str
    label: str


class _FakeDataset:
    """Mimics the shared corpus-recipe Dataset shape used across egs3/."""

    def __init__(self, entries_by_split):
        self._entries_by_split = entries_by_split

    def __call__(self, split):
        self._entries = self._entries_by_split[split]
        return self


def _fake_module(entries_by_split):
    class _Module:
        pass

    module = _Module()
    module.Dataset = _FakeDataset(entries_by_split)
    return module


def _patch_corpora(monkeypatch, corpora_entries):
    """corpora_entries: {data_src: {split: [entries]}}"""

    def fake_load_dataset_module(data_src=None, recipe_dir=None):
        return _fake_module(corpora_entries[data_src])

    monkeypatch.setattr(mm, "load_dataset_module", fake_load_dataset_module)


def test_merge_applies_label_map_and_drops(monkeypatch):
    corpora_entries = {
        "a/esp2_cls": {
            "train": [
                _FakeEntry("a-1", "/a/1.wav", "anger"),
                _FakeEntry("a-2", "/a/2.wav", "boredom"),
            ]
        },
        "b/esp2_cls": {
            "train": [
                _FakeEntry("b-1", "/b/1.wav", "angry"),
            ]
        },
    }
    _patch_corpora(monkeypatch, corpora_entries)

    result = mm.merge_train_valid_manifests(
        [mm.CorpusRef("a", "a/esp2_cls"), mm.CorpusRef("b", "b/esp2_cls")],
        label_map={"anger": "angry", "angry": "angry"},
        splits=("train",),
    )

    merged = result["train"]
    assert merged.lines == [
        "a-1\t/a/1.wav\tangry\n",
        "b-1\t/b/1.wav\tangry\n",
    ]
    assert merged.kept[("a", "angry")] == 1
    assert merged.kept[("b", "angry")] == 1
    assert merged.dropped[("a", "boredom")] == 1


def test_merge_sorts_and_logs_per_corpus(monkeypatch):
    corpora_entries = {
        "a/esp2_cls": {
            "train": [
                _FakeEntry("a-z", "/a/z.wav", "angry"),
                _FakeEntry("a-a", "/a/a.wav", "angry"),
            ]
        },
    }
    _patch_corpora(monkeypatch, corpora_entries)

    result = mm.merge_train_valid_manifests(
        [mm.CorpusRef("a", "a/esp2_cls")],
        label_map={"angry": "angry"},
        splits=("train",),
    )

    assert [line.split("\t")[0] for line in result["train"].lines] == ["a-a", "a-z"]


def test_merge_duplicate_utt_id_raises(monkeypatch):
    corpora_entries = {
        "a/esp2_cls": {"train": [_FakeEntry("dup-1", "/a/1.wav", "angry")]},
        "b/esp2_cls": {"train": [_FakeEntry("dup-1", "/b/1.wav", "angry")]},
    }
    _patch_corpora(monkeypatch, corpora_entries)

    with pytest.raises(RuntimeError, match="Duplicate utterance id"):
        mm.merge_train_valid_manifests(
            [mm.CorpusRef("a", "a/esp2_cls"), mm.CorpusRef("b", "b/esp2_cls")],
            label_map={"angry": "angry"},
            splits=("train",),
        )


def test_build_eval_manifest_single_corpus(monkeypatch):
    corpora_entries = {
        "msppodcast/esp2_cls": {
            "test": [
                _FakeEntry("msppodcast-1", "/msp/1.wav", "angry"),
                _FakeEntry("msppodcast-2", "/msp/2.wav", "contempt"),
            ]
        },
    }
    _patch_corpora(monkeypatch, corpora_entries)

    merged = mm.build_eval_manifest(
        mm.CorpusRef("msppodcast", "msppodcast/esp2_cls"),
        split="test",
        label_map={"angry": "angry"},
    )

    assert merged.lines == ["msppodcast-1\t/msp/1.wav\tangry\n"]
    assert merged.dropped[("msppodcast", "contempt")] == 1


def test_write_manifest_atomic(tmp_path: Path):
    manifest_path = tmp_path / "manifest" / "train.tsv"
    mm.write_manifest(["a-1\t/a/1.wav\tangry\n"], manifest_path)

    assert manifest_path.is_file()
    assert manifest_path.read_text(encoding="utf-8") == "a-1\t/a/1.wav\tangry\n"
    assert not manifest_path.with_suffix(".tsv.part").exists()
