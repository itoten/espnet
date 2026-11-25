"""sinhala-emotional-tts-dataset dataset module."""

from egs3.sinhala_emotional_tts_dataset.cls.dataset.builder import SinhalaEmotionalTTSBuilder as DatasetBuilder
from egs3.sinhala_emotional_tts_dataset.cls.dataset.dataset import SinhalaEmotionalTTSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
