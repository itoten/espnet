"""Multi-corpus (core7) dataset module."""

from egs3.multicorpus_core7.cls.dataset.builder import (
    MulticorpusBuilder as DatasetBuilder,
)
from egs3.multicorpus_core7.cls.dataset.dataset import MulticorpusDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
