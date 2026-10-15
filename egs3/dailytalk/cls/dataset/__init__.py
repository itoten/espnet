"""DailyTalk dataset module."""

from egs3.dailytalk.cls.dataset.builder import DailytalkBuilder as DatasetBuilder
from egs3.dailytalk.cls.dataset.dataset import DailytalkDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
