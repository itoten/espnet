"""LEGOv2 dataset module."""

from egs3.legov2.cls.dataset.builder import Legov2Builder as DatasetBuilder
from egs3.legov2.cls.dataset.dataset import Legov2Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
