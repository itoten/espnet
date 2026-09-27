"""EmoV-DB dataset module."""

from egs3.emovdb.wavlmasp.dataset.builder import EmoVDBBuilder as DatasetBuilder
from egs3.emovdb.wavlmasp.dataset.dataset import EmoVDBDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
