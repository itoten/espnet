"""IEMOCAP dataset module."""

from egs3.iemocap.cls.dataset.builder import IEMOCAPBuilder as DatasetBuilder
from egs3.iemocap.cls.dataset.dataset import IEMOCAPDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
