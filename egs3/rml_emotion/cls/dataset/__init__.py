"""RML Emotion Database dataset module."""

from egs3.rml_emotion.cls.dataset.builder import RmlEmotionBuilder as DatasetBuilder
from egs3.rml_emotion.cls.dataset.dataset import RmlEmotionDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
