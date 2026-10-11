"""EMOVO dataset module."""

from egs3.emovo.cls.dataset.builder import EMOVOBuilder as DatasetBuilder
from egs3.emovo.cls.dataset.dataset import EMOVODataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
