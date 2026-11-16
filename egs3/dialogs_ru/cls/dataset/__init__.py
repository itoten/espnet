"""Dialogs dataset module."""

from egs3.dialogs_ru.cls.dataset.builder import DialogsRuBuilder as DatasetBuilder
from egs3.dialogs_ru.cls.dataset.dataset import DialogsRuDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
