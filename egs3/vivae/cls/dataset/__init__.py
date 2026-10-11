"""VIVAE dataset module."""

from egs3.vivae.cls.dataset.builder import VIVAEBuilder as DatasetBuilder
from egs3.vivae.cls.dataset.dataset import VIVAEDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
