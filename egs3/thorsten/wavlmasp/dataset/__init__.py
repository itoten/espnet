"""Thorsten-Emotional dataset module."""

from egs3.thorsten.wavlmasp.dataset.builder import ThorstenBuilder as DatasetBuilder
from egs3.thorsten.wavlmasp.dataset.dataset import ThorstenDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
