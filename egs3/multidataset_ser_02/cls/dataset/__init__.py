"""Multi-dataset (multidataset_ser_02) dataset module."""

from egs3.multidataset_ser_02.cls.dataset.builder import (
    MultidatasetSer02Builder as DatasetBuilder,
)
from egs3.multidataset_ser_02.cls.dataset.dataset import (
    MultidatasetSer02Dataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
