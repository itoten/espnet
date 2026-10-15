"""CASIA dataset module."""

from egs3.casia.cls.dataset.builder import CasiaBuilder as DatasetBuilder
from egs3.casia.cls.dataset.dataset import CasiaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
