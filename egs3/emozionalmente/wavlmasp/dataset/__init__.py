"""Single-corpus dataset module."""

from egs3.emozionalmente.wavlmasp.dataset.builder import (
    MulticorpusBuilder as DatasetBuilder,
)
from egs3.emozionalmente.wavlmasp.dataset.dataset import MulticorpusDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
