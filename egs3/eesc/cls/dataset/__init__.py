"""EESC dataset module."""

from egs3.eesc.cls.dataset.builder import EescBuilder as DatasetBuilder
from egs3.eesc.cls.dataset.dataset import EescDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
