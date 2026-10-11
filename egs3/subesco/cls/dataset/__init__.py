"""SUBESCO dataset module."""

from egs3.subesco.cls.dataset.builder import SUBESCOBuilder as DatasetBuilder
from egs3.subesco.cls.dataset.dataset import SUBESCODataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
