"""OpenBMI dataset module."""

from egs3.openbmi.eegcls.dataset.builder import OpenBMIBuilder as DatasetBuilder
from egs3.openbmi.eegcls.dataset.dataset import OpenBMIDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
