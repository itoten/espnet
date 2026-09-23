"""Multi-corpus (core7) dataset module."""

from egs3.multicorpus_full20.wavlmasp.dataset.builder import (
    MulticorpusBuilder as DatasetBuilder,
)
from egs3.multicorpus_full20.wavlmasp.dataset.dataset import (
    MulticorpusDataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
