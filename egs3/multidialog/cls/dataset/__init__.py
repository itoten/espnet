"""MultiDialog dataset module."""

from egs3.multidialog.cls.dataset.builder import MultidialogBuilder as DatasetBuilder
from egs3.multidialog.cls.dataset.dataset import MultidialogDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
