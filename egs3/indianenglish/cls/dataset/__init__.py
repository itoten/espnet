"""Indian English dataset module."""

from egs3.indianenglish.cls.dataset.builder import (
    IndianEnglishBuilder as DatasetBuilder,
)
from egs3.indianenglish.cls.dataset.dataset import IndianEnglishDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
