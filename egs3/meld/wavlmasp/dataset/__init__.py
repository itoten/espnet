"""MELD dataset module."""

from egs3.meld.wavlmasp.dataset.builder import MELDBuilder as DatasetBuilder
from egs3.meld.wavlmasp.dataset.dataset import MELDDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
