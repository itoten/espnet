"""EmoTa dataset module."""

from egs3.emota.cls.dataset.builder import EmoTaBuilder as DatasetBuilder
from egs3.emota.cls.dataset.dataset import EmoTaDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
