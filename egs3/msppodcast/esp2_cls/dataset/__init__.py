"""MSP-Podcast dataset module."""

from egs3.msppodcast.esp2_cls.dataset.builder import MSPPodcastBuilder as DatasetBuilder
from egs3.msppodcast.esp2_cls.dataset.dataset import MSPPodcastDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
