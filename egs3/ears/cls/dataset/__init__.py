"""EARS dataset module."""

from egs3.ears.cls.dataset.builder import EARSBuilder as DatasetBuilder
from egs3.ears.cls.dataset.dataset import EARSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
