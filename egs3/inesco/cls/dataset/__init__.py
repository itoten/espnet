"""INESCO dataset module."""

from egs3.inesco.cls.dataset.builder import InescoBuilder as DatasetBuilder
from egs3.inesco.cls.dataset.dataset import InescoDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
