"""Dataset module of multidataset_ser_01/esp2_cls: the `cls` recipe's, as is.

The manifests, the label list and the WAV files are those that
`egs3/multidataset_ser_01/cls` prepares; this recipe only reads them.
"""

from egs3.multidataset_ser_01.cls.dataset import Dataset, DatasetBuilder

__all__ = ["Dataset", "DatasetBuilder"]
