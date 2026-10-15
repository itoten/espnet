"""UTeMo dataset module."""

from egs3.utemo.cls.dataset.builder import UtemoBuilder as DatasetBuilder
from egs3.utemo.cls.dataset.dataset import UtemoDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
