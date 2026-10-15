"""ReCANVo dataset module."""

from egs3.recanvo.cls.dataset.builder import RecanvoBuilder as DatasetBuilder
from egs3.recanvo.cls.dataset.dataset import RecanvoDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
