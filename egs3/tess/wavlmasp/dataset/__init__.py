"""TESS dataset module."""

from egs3.tess.wavlmasp.dataset.builder import TESSBuilder as DatasetBuilder
from egs3.tess.wavlmasp.dataset.dataset import TESSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
