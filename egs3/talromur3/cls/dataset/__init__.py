"""Talromur 3 dataset module."""

from egs3.talromur3.cls.dataset.builder import Talromur3Builder as DatasetBuilder
from egs3.talromur3.cls.dataset.dataset import Talromur3Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
