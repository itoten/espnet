"""BanglaSER dataset module."""

from egs3.banglaser.cls.dataset.builder import BanglaSERBuilder as DatasetBuilder
from egs3.banglaser.cls.dataset.dataset import BanglaSERDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
