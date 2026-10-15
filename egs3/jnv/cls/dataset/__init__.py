"""JNV dataset module."""

from egs3.jnv.cls.dataset.builder import JnvBuilder as DatasetBuilder
from egs3.jnv.cls.dataset.dataset import JnvDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
