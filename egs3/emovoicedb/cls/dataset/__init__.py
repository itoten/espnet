"""EmoVoice-DB dataset module."""

from egs3.emovoicedb.cls.dataset.builder import EmovoicedbBuilder as DatasetBuilder
from egs3.emovoicedb.cls.dataset.dataset import EmovoicedbDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
