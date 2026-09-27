"""MSP-Podcast dataset module."""

from egs3.msppodcast.wavlmasp.dataset.builder import MSPPodcastBuilder as DatasetBuilder
from egs3.msppodcast.wavlmasp.dataset.dataset import MSPPodcastDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
