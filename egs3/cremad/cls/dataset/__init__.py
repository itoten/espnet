"""CREMA-D dataset module."""

from egs3.cremad.cls.dataset.builder import CremaDBuilder as DatasetBuilder
from egs3.cremad.cls.dataset.dataset import CremaDDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
