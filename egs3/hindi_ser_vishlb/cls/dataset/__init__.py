"""Hindi SER (vishlb) dataset module."""

from egs3.hindi_ser_vishlb.cls.dataset.builder import (
    HindiSerVishlbBuilder as DatasetBuilder,
)
from egs3.hindi_ser_vishlb.cls.dataset.dataset import HindiSerVishlbDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
