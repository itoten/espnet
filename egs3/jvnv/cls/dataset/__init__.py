"""JVNV dataset module."""

from egs3.jvnv.cls.dataset.builder import JVNVBuilder as DatasetBuilder
from egs3.jvnv.cls.dataset.dataset import JVNVDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
