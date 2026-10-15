"""EmoTale dataset module."""

from egs3.emotale.cls.dataset.builder import EmotaleBuilder as DatasetBuilder
from egs3.emotale.cls.dataset.dataset import EmotaleDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
