"""PAVOQUE dataset module."""

from egs3.pavoque.cls.dataset.builder import PavoqueBuilder as DatasetBuilder
from egs3.pavoque.cls.dataset.dataset import PavoqueDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
