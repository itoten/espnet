"""Thanushs25 Tamil dataset module."""

from egs3.thanushs25_tamil.cls.dataset.builder import (
    Thanushs25TamilBuilder as DatasetBuilder,
)
from egs3.thanushs25_tamil.cls.dataset.dataset import Thanushs25TamilDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
