"""CSEMOTIONS dataset module."""

from egs3.csemotions.cls.dataset.builder import CsemotionsBuilder as DatasetBuilder
from egs3.csemotions.cls.dataset.dataset import CsemotionsDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
