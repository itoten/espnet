"""EMIS dataset module."""

from egs3.emis.cls.dataset.builder import EmisBuilder as DatasetBuilder
from egs3.emis.cls.dataset.dataset import EmisDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
