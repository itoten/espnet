"""KazEmoTTS dataset module."""

from egs3.kazemotts.cls.dataset.builder import KazEmoTTSBuilder as DatasetBuilder
from egs3.kazemotts.cls.dataset.dataset import KazEmoTTSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
