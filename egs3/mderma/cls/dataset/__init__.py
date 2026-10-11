"""MDER-MA dataset module."""

from egs3.mderma.cls.dataset.builder import MdermaBuilder as DatasetBuilder
from egs3.mderma.cls.dataset.dataset import MdermaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
