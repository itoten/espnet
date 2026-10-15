"""E-SERAVD dataset module."""

from egs3.eseravd.cls.dataset.builder import EseravdBuilder as DatasetBuilder
from egs3.eseravd.cls.dataset.dataset import EseravdDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
