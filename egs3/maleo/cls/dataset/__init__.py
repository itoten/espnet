"""Maleo dataset module."""

from egs3.maleo.cls.dataset.builder import MaleoBuilder as DatasetBuilder
from egs3.maleo.cls.dataset.dataset import MaleoDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
