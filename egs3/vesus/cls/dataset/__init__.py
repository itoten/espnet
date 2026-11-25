"""VESUS dataset module."""

from egs3.vesus.cls.dataset.builder import VESUSBuilder as DatasetBuilder
from egs3.vesus.cls.dataset.dataset import VESUSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
