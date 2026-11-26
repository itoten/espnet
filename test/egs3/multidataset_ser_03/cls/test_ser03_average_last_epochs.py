import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[4] / "egs3/multidataset_ser_03/cls/local/average_last_epochs.py"


def _load():
    spec = importlib.util.spec_from_file_location("average_last_epochs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_averages_the_last_epochs(tmp_path, monkeypatch):
    (tmp_path / "epochs").mkdir()
    for epoch in (1, 3, 5, 7, 9, 11):
        state = {
            "model.w": torch.full((2,), float(epoch)),
            "model.n": torch.tensor(epoch, dtype=torch.int64),
        }
        torch.save({"state_dict": state}, tmp_path / "epochs" / f"epoch{epoch}.ckpt")
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), "--num", "3"])
    _load().main()
    average = torch.load(tmp_path / "epochs.ave_last3.pth")
    # epochs 7, 9 and 11 (sorted by number, not as text), without `model.`
    torch.testing.assert_close(average["w"], torch.full((2,), 9.0))
    assert set(average) == {"w", "n"}
    assert not (tmp_path / "epochs.ave_3best.pth").exists()
