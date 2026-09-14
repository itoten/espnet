"""JL-Corpus dataset module."""

from egs3.jlcorpus.cls.dataset.builder import JLCorpusBuilder as DatasetBuilder
from egs3.jlcorpus.cls.dataset.dataset import JLCorpusDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
