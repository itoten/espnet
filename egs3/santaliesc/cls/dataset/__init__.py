"""SantaliESC dataset module."""

from egs3.santaliesc.cls.dataset.builder import SantaliESCBuilder as DatasetBuilder
from egs3.santaliesc.cls.dataset.dataset import SantaliESCDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
