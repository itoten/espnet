"""AkaCE (Akan Cinematic Emotions) dataset module."""

from egs3.akace.cls.dataset.builder import AkaceBuilder as DatasetBuilder
from egs3.akace.cls.dataset.dataset import AkaceDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
