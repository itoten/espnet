"""emoUERJ dataset module."""

from egs3.emouerj.cls.dataset.builder import EmoUERJBuilder as DatasetBuilder
from egs3.emouerj.cls.dataset.dataset import EmoUERJDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
