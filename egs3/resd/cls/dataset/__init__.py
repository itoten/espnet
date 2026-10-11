"""RESD dataset module."""

from egs3.resd.cls.dataset.builder import RESDBuilder as DatasetBuilder
from egs3.resd.cls.dataset.dataset import RESDDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
