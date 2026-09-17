"""MSP-Podcast dataset module."""

from egs3.msppodcast.cls.dataset.builder import MSPPodcastBuilder as DatasetBuilder
from egs3.msppodcast.cls.dataset.dataset import MSPPodcastDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
