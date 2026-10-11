"""IndoWaveSentiment dataset module."""

from egs3.indowavesentiment.cls.dataset.builder import (
    IndoWaveSentimentBuilder as DatasetBuilder,
)
from egs3.indowavesentiment.cls.dataset.dataset import (
    IndoWaveSentimentDataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
