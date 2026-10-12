"""UrduSER dataset module."""

from egs3.urduser.cls.dataset.builder import UrduserBuilder as DatasetBuilder
from egs3.urduser.cls.dataset.dataset import UrduserDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
