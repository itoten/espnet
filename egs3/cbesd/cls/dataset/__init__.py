"""C-BESD dataset module."""

from egs3.cbesd.cls.dataset.builder import CbesdBuilder as DatasetBuilder
from egs3.cbesd.cls.dataset.dataset import CbesdDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
