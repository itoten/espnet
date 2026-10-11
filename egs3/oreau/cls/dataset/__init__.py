"""Oréau dataset module."""

from egs3.oreau.cls.dataset.builder import OreauBuilder as DatasetBuilder
from egs3.oreau.cls.dataset.dataset import OreauDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
