"""Thorsten-Emotional dataset module."""

from egs3.thorsten.cls.dataset.builder import ThorstenBuilder as DatasetBuilder
from egs3.thorsten.cls.dataset.dataset import ThorstenDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
