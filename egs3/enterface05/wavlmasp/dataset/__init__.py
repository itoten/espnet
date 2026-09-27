"""eNTERFACE05 dataset module."""

from egs3.enterface05.wavlmasp.dataset.builder import (
    ENTERFACE05Builder as DatasetBuilder,
)
from egs3.enterface05.wavlmasp.dataset.dataset import ENTERFACE05Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
