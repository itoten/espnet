"""YSED dataset module."""

from egs3.ysed.cls.dataset.builder import YSEDBuilder as DatasetBuilder
from egs3.ysed.cls.dataset.dataset import YSEDDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
