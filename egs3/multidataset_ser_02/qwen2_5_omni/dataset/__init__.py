"""Evaluation data of multidataset_ser_02 for prompting a speech LLM.

The data is prepared by the `cls` recipe (`egs3/multidataset_ser_02/cls`,
stages `create_dataset` and `remove_long_short`); this recipe only reads it,
so `DatasetBuilder` is that recipe's builder.
"""

from egs3.multidataset_ser_02.cls.dataset.builder import (
    MultidatasetSer02Builder as DatasetBuilder,
)
from egs3.multidataset_ser_02.qwen2_5_omni.dataset.dataset import (
    VoxEmoEvalDataset as Dataset,
)

__all__ = ["Dataset", "DatasetBuilder"]
