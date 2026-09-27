"""JL-Corpus dataset module."""

from egs3.jlcorpus.wavlmasp.dataset.builder import JLCorpusBuilder as DatasetBuilder
from egs3.jlcorpus.wavlmasp.dataset.dataset import JLCorpusDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
