"""BEAT dataset module."""

from egs3.beat.cls.dataset.builder import BeatBuilder as DatasetBuilder
from egs3.beat.cls.dataset.dataset import BeatDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
