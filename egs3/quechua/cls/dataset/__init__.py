"""Quechua Collao dataset module."""

from egs3.quechua.cls.dataset.builder import QuechuaBuilder as DatasetBuilder
from egs3.quechua.cls.dataset.dataset import QuechuaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
