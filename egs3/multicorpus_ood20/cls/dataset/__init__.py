"""Multi-corpus (ood20) dataset module."""

from egs3.multicorpus_ood20.cls.dataset.builder import (
    MulticorpusBuilder as DatasetBuilder,
)
from egs3.multicorpus_ood20.cls.dataset.dataset import MulticorpusDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
