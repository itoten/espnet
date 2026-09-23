"""MSP-Podcast-only dataset module."""

from egs3.msppodcast.wavlmasp.dataset.builder import (
    MulticorpusBuilder as DatasetBuilder,
)
from egs3.msppodcast.wavlmasp.dataset.dataset import MulticorpusDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
