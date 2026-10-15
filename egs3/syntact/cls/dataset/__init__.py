"""SyntAct dataset module."""

from egs3.syntact.cls.dataset.builder import SyntactBuilder as DatasetBuilder
from egs3.syntact.cls.dataset.dataset import SyntactDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
