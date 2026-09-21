"""EMNS dataset module."""

from egs3.emns.esp2_cls.dataset.builder import EMNSBuilder as DatasetBuilder
from egs3.emns.esp2_cls.dataset.dataset import EMNSDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
