"""Single-corpus dataset module."""

from egs3.jlcorpus.wavlmasp.dataset.builder import MulticorpusBuilder as DatasetBuilder
from egs3.jlcorpus.wavlmasp.dataset.dataset import MulticorpusDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
