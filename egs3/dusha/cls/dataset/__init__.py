"""Dusha dataset module."""

from egs3.dusha.cls.dataset.builder import DushaBuilder as DatasetBuilder
from egs3.dusha.cls.dataset.dataset import DushaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
