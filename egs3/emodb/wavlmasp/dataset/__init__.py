"""EmoDB dataset module."""

from egs3.emodb.wavlmasp.dataset.builder import EmoDBBuilder as DatasetBuilder
from egs3.emodb.wavlmasp.dataset.dataset import EmoDBDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
