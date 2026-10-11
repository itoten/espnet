"""KBES dataset module."""

from egs3.kbes.cls.dataset.builder import KBESBuilder as DatasetBuilder
from egs3.kbes.cls.dataset.dataset import KBESDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
