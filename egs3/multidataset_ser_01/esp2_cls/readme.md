# Multi-dataset SER with a log-mel Transformer trained from scratch (multidataset_ser_01/esp2_cls)

The `esp2_cls` system (ESPnet2's classification task) on the data of
`egs3/multidataset_ser_01/cls`: 80-bin log-mel features, SpecAugment, a
Transformer encoder trained from scratch, mean pooling and a
linear layer to the seven classes. No pre-trained model is used.

## Data

Everything about the data is in `../cls/readme.md`. This recipe reads what
that recipe prepares and prepares nothing itself:

- training and validation: `manifest_filtered/{train,valid}.tsv` (0.5 to 30 s),
- evaluation: `manifest/<corpus>.tsv` for each of the 26 evaluation corpora,
- the label list: `token_list`,

all under `$MULTIDATASET_SER_01_OUTPUT` (default `../cls/data`). The dataset
module is `../cls`'s, so the two recipes see the same rows.

## Model and training

In `conf/training.yaml`: a Transformer encoder of 12 blocks, 256 dims and 4
heads, after a convolutional front layer that subsamples the 100 frames per
second by 4 (`conv2d`); AdamW (weight decay 0.01) with a peak lr of 1e-3 and
TristageLR (warm-up over the first 10 % of the updates, hold for 40 %, decay
over the last 50 %; `max_steps` is set for 8 GPUs), batch size 32 per GPU,
features normalised with the training set's mean and variance
(`collect_stats`), the best checkpoint by validation accuracy kept
(`valid.acc.ave_1best.pth`) and the weights of every epoch under
`exp/<tag>/epochs/`, training stopped after 10 epochs without improvement (at
most 40). These values are not tuned yet.

## Running

Prepare the data in `../cls` first (its `create_dataset`,
`remove_long_short` and `prepare_labels` stages). Then, here, run each line
as its own command:

```bash
python run.py --stages collect_stats --training_config conf/training.yaml
python run.py --stages train --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

Keep `collect_stats` and `train` in separate commands: run together,
`collect_stats` removes `normalize` from the configuration and training falls
back to per-utterance normalisation without saying so.
Results go to `exp/train_transformer/inference/`: WA, UA, macro F1, mAP and AUC
per evaluation corpus, as in `../cls`.

Environment variables:

- `NUM_DEVICE`: GPUs to train on (1 by default; DDP when more).
- `MULTIDATASET_SER_01_WORKERS`: worker processes for `collect_stats`.
- `RESUME_CKPT`: a checkpoint (e.g. `exp/train_transformer/last.ckpt`) to
  resume training from.
- `MULTIDATASET_SER_01_EXP_TAG`: the directory under `exp/` to write to
  (`train_transformer` by default); give the same value to `infer measure`.
