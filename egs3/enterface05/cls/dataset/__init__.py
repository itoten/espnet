"""eNTERFACE05 dataset module."""

from egs3.enterface05.cls.dataset.builder import ENTERFACE05Builder as DatasetBuilder
from egs3.enterface05.cls.dataset.dataset import ENTERFACE05Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
