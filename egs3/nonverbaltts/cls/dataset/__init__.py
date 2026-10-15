"""NonverbalTTS dataset module."""

from egs3.nonverbaltts.cls.dataset.builder import NonverbalttsBuilder as DatasetBuilder
from egs3.nonverbaltts.cls.dataset.dataset import NonverbalttsDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
