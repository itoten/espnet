"""Thorsten-Emotional dataset module."""

from egs3.thorsten.esp2_cls.dataset.builder import ThorstenBuilder as DatasetBuilder
from egs3.thorsten.esp2_cls.dataset.dataset import ThorstenDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
