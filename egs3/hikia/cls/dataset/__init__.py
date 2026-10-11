"""HiKIA dataset module."""

from egs3.hikia.cls.dataset.builder import HiKIABuilder as DatasetBuilder
from egs3.hikia.cls.dataset.dataset import HiKIADataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
