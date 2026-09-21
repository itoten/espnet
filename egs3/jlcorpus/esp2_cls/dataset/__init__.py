"""JL-Corpus dataset module."""

from egs3.jlcorpus.esp2_cls.dataset.builder import JLCorpusBuilder as DatasetBuilder
from egs3.jlcorpus.esp2_cls.dataset.dataset import JLCorpusDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
