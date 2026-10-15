"""SITB-OSED (Indic-Odia Speech Emotion Data) dataset module."""

from egs3.sitb_osed.cls.dataset.builder import SitbOsedBuilder as DatasetBuilder
from egs3.sitb_osed.cls.dataset.dataset import SitbOsedDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
