"""SES-ED dataset module."""

from egs3.ses_ed.cls.dataset.builder import SesEdBuilder as DatasetBuilder
from egs3.ses_ed.cls.dataset.dataset import SesEdDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
