"""Emo-Emilia dataset module."""

from egs3.emo_emilia.cls.dataset.builder import EmoEmiliaBuilder as DatasetBuilder
from egs3.emo_emilia.cls.dataset.dataset import EmoEmiliaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
