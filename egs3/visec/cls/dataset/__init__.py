"""ViSEC dataset module."""

from egs3.visec.cls.dataset.builder import VisecBuilder as DatasetBuilder
from egs3.visec.cls.dataset.dataset import VisecDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
