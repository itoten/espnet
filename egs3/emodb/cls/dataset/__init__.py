"""EmoDB dataset module."""

from egs3.emodb.cls.dataset.builder import EmoDBBuilder as DatasetBuilder
from egs3.emodb.cls.dataset.dataset import EmoDBDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
