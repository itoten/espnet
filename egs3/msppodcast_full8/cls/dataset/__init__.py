"""MSP-Podcast (full 8-class) dataset module."""

from egs3.msppodcast_full8.cls.dataset.builder import (
    MSPPodcastFull8Builder as DatasetBuilder,
)
from egs3.msppodcast_full8.cls.dataset.dataset import MSPPodcastFull8Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
