"""EMNS dataset module."""

from egs3.emns.cls.dataset.builder import EMNSBuilder as DatasetBuilder
from egs3.emns.cls.dataset.dataset import EMNSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
