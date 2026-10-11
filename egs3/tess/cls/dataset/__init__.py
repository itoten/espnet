"""TESS dataset module."""

from egs3.tess.cls.dataset.builder import TESSBuilder as DatasetBuilder
from egs3.tess.cls.dataset.dataset import TESSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
