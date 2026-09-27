"""IEMOCAP dataset module."""

from egs3.iemocap.wavlmasp.dataset.builder import IEMOCAPBuilder as DatasetBuilder
from egs3.iemocap.wavlmasp.dataset.dataset import IEMOCAPDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
