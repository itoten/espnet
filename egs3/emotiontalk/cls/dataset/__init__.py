"""EmotionTalk dataset module."""

from egs3.emotiontalk.cls.dataset.builder import EmotiontalkBuilder as DatasetBuilder
from egs3.emotiontalk.cls.dataset.dataset import EmotiontalkDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
