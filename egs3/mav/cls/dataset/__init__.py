"""MAV dataset module."""

from egs3.mav.cls.dataset.builder import MavBuilder as DatasetBuilder
from egs3.mav.cls.dataset.dataset import MavDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
