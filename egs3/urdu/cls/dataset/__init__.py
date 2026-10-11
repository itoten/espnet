"""URDU dataset module."""

from egs3.urdu.cls.dataset.builder import URDUBuilder as DatasetBuilder
from egs3.urdu.cls.dataset.dataset import URDUDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
