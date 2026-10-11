"""Kannada dataset module."""

from egs3.kannada.cls.dataset.builder import KannadaBuilder as DatasetBuilder
from egs3.kannada.cls.dataset.dataset import KannadaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
