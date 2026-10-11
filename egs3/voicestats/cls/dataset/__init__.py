"""Voice Statistics corpus dataset module."""

from egs3.voicestats.cls.dataset.builder import VoiceStatsBuilder as DatasetBuilder
from egs3.voicestats.cls.dataset.dataset import VoiceStatsDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
