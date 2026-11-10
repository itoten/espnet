"""Dataset module of multidataset_ser_01/esp2_lid (data prepared by `../cls`)."""

from egs3.multidataset_ser_01.cls.dataset import DatasetBuilder
from egs3.multidataset_ser_01.esp2_lid.dataset.dataset import (
    MultidatasetSer01LIDDataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
