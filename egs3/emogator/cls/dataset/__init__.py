"""EmoGator dataset module."""

from egs3.emogator.cls.dataset.builder import EmogatorBuilder as DatasetBuilder
from egs3.emogator.cls.dataset.dataset import EmogatorDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
