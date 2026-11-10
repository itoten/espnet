"""Dataset of multidataset_ser_01/esp2_lid: `../cls`'s, with the label as `lid_labels`.

The `esp2_lid` system reads the class of an utterance from the `lid_labels`
field. The manifests and WAV files are those `egs3/multidataset_ser_01/cls`
prepares; this module only renames the field.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import soundfile as sf

from egs3.multidataset_ser_01.cls.dataset.dataset import MultidatasetSer01Dataset
from espnet3.contract import Field


class MultidatasetSer01LIDDataset(MultidatasetSer01Dataset):
    """Merged dataset that returns ``{"speech", "lid_labels"}`` samples.

    Takes the same arguments as
    :class:`egs3.multidataset_ser_01.cls.dataset.dataset.MultidatasetSer01Dataset`.
    """

    fields = (Field("speech", "audio"), Field("lid_labels", "text"))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Not `super().__getitem__`: `BaseDataset` checks every `__getitem__`'s
        # output against this class's `fields`, which have no `label`.
        entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "lid_labels": entry.label,
        }
