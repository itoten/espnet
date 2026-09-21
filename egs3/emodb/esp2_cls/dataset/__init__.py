"""EmoDB dataset module."""

from egs3.emodb.esp2_cls.dataset.builder import EmoDBBuilder as DatasetBuilder
from egs3.emodb.esp2_cls.dataset.dataset import EmoDBDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
