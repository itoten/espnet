"""MaSERC dataset module."""

from egs3.maserc.cls.dataset.builder import MasercBuilder as DatasetBuilder
from egs3.maserc.cls.dataset.dataset import MasercDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
