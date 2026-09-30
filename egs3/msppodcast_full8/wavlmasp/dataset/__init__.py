"""MSP-Podcast (full 8-class) dataset module."""

from egs3.msppodcast_full8.wavlmasp.dataset.builder import (
    MSPPodcastFull8Builder as DatasetBuilder,
)
from egs3.msppodcast_full8.wavlmasp.dataset.dataset import (
    MSPPodcastFull8Dataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
