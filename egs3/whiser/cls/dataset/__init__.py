"""WHiSER dataset module."""

from egs3.whiser.cls.dataset.builder import WhiserBuilder as DatasetBuilder
from egs3.whiser.cls.dataset.dataset import WhiserDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
