"""Telugu Emotion Speech dataset module."""

from egs3.telugu_emotion_kaggle.cls.dataset.builder import (
    TeluguEmotionKaggleBuilder as DatasetBuilder,
)
from egs3.telugu_emotion_kaggle.cls.dataset.dataset import (
    TeluguEmotionKaggleDataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
