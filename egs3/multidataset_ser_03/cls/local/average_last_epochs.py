"""Average the weights of the last validated epochs of a run.

Training keeps the weights of every validated epoch under
``<exp_dir>/epochs/epoch<N>.ckpt`` (see ``conf/training.yaml``). This writes
the average of the last ``--num`` of them to
``<exp_dir>/epochs.ave_last<num>.pth``, which ``conf/inference.yaml`` scores
with ``MULTIDATASET_SER_03_WEIGHTS=epochs.ave_last<num>``. The average is
taken by ESPnet3's own ``AverageCheckpointsCallback`` (the one that writes
``valid.acc.ave_<K>best.pth`` during training), given these checkpoints
instead of the best ones, as ESPnet1's ``utils/average_checkpoints.py``
averages the last snapshots of a run.

Usage:
    python local/average_last_epochs.py <exp_dir> [--num 5]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

from espnet2.torch_utils.safe_torch_load import safe_torch_load
from espnet3.components.callbacks.default_callbacks import AverageCheckpointsCallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    epochs = sorted(
        (args.exp_dir / "epochs").glob("epoch*.ckpt"),
        key=lambda p: int(re.fullmatch(r"epoch(\d+)\.ckpt", p.name).group(1)),
    )
    if len(epochs) < args.num:
        raise SystemExit(f"{len(epochs)} epoch checkpoints, fewer than --num {args.num}")
    last = epochs[-args.num :]
    print("averaging", [p.name for p in last])

    # The callback checks the averaged keys against the model's own keys;
    # the model is not built here, so they are taken from a checkpoint.
    state = safe_torch_load(last[0], map_location="cpu")
    state = state.get("state_dict", state)
    keys = {k.removeprefix("model.") for k in state}
    model = SimpleNamespace(state_dict=lambda: dict.fromkeys(keys))
    checkpoints = SimpleNamespace(monitor="epochs", best_k_models={str(p): None for p in last})
    callback = AverageCheckpointsCallback(args.exp_dir, [checkpoints])
    callback._write_averages(SimpleNamespace(is_global_zero=True), model)

    written = args.exp_dir / f"epochs.ave_{args.num}best.pth"
    target = args.exp_dir / f"epochs.ave_last{args.num}.pth"
    written.rename(target)
    print("wrote", target)


if __name__ == "__main__":
    main()
