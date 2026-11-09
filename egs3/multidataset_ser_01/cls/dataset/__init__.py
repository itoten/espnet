"""Multi-dataset (multidataset_ser_01) dataset module."""

from egs3.multidataset_ser_01.cls.dataset.builder import (
    MultidatasetSer01Builder as DatasetBuilder,
)
from egs3.multidataset_ser_01.cls.dataset.dataset import (
    MultidatasetSer01Dataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
