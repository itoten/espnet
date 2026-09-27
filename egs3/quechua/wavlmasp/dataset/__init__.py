"""Quechua Collao dataset module."""

from egs3.quechua.wavlmasp.dataset.builder import QuechuaBuilder as DatasetBuilder
from egs3.quechua.wavlmasp.dataset.dataset import QuechuaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
