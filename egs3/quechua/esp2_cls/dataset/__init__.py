"""Quechua Collao dataset module."""

from egs3.quechua.esp2_cls.dataset.builder import QuechuaBuilder as DatasetBuilder
from egs3.quechua.esp2_cls.dataset.dataset import QuechuaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
