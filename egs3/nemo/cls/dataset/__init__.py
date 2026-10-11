"""nEMO dataset module."""

from egs3.nemo.cls.dataset.builder import NEMOBuilder as DatasetBuilder
from egs3.nemo.cls.dataset.dataset import NEMODataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
