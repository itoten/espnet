# Multi-dataset SER with ECAPA-TDNN trained from scratch (multidataset_ser_01/esp2_lid)

The `esp2_lid` system (ESPnet2's language identification task) on the data of
`egs3/multidataset_ser_01/cls`, with the seven emotions in place of
languages: 80-bin log-mel features, an ECAPA-TDNN encoder, channel- and
context-dependent attentive statistics pooling, a 192-dim embedding, and an
additive angular margin softmax with sub-centres. No pre-trained model is
used.

## Data

Everything about the data is in `../cls/readme.md`. This recipe reads what
that recipe prepares and prepares nothing itself:

- training and validation: `manifest_filtered/{train,valid}.tsv` (0.5 to 30 s),
- evaluation: `manifest_filtered/<corpus>.tsv` for each of the 26 evaluation
  corpora (0.5 to 30 s as well),

all under `$MULTIDATASET_SER_01_OUTPUT` (default `../cls/data`). The dataset
(`dataset/dataset.py`) is `../cls`'s with the label returned as `lid_labels`,
the field the `esp2_lid` system reads. `../cls`'s `token_list` is not used:
`collect_stats` lists the classes itself (`exp/stats/train/lang2utt`, in
alphabetical order).

## Model and training

In `conf/training.yaml`; provisional and not tuned:

- ECAPA-TDNN with the sizes of ESPnet2's VoxCeleb recipe (C = 1024,
  1536-dim output), RawNet3 projector to 192 dims.
- Loss: AAM-softmax with 3 sub-centres per class, margin 0.2, scale 30.
  The inter-top-k penalty is off (`k_top: 0`): with seven classes it fails
  for `k_top` of 6 or more, and gives no benefit at that size.
- Batches: the `catpow` sampler draws utterances class by class, with each
  class's share set by its total duration raised to `upsampling_factor`
  (0.5); about 200 s of audio per batch (padding not counted), at most 64
  utterances. An epoch covers
  the training audio about once. Validation uses every validation utterance
  once (`sorted`).
- No SpecAugment.
- AdamW (weight decay 0.01) with a peak lr of 1e-3 and TristageLR (warm-up
  over the first 10 % of the updates, hold for 40 %, decay over the last
  50 %) over 39,160 updates (40 epochs of the 443 h of training audio on 8
  GPUs; recompute `max_steps` when the batch size, the number of GPUs or
  `max_epochs` changes). The best checkpoint by
  validation accuracy is kept (`valid.accuracy.ave_1best.pth`) and the
  weights of every epoch under `exp/<tag>/epochs/`; training stops after 10
  epochs without improvement (at most 40).

Inference writes the predicted class only, with no per-class scores, so the
metrics are accuracy (WA) and a classification report whose macro-averaged
recall is UA; `../cls`'s mAP and AUC are not available.

## Running

Prepare the data in `../cls` first (its `create_dataset` and
`remove_long_short` stages). Then, here, run each line as its own command:

```bash
python run.py --stages collect_stats --training_config conf/training.yaml
python run.py --stages train --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`collect_stats` reads every training and validation file once. Results go to
`exp/train_ecapa_tdnn/inference/`.

Environment variables:

- `NUM_DEVICE`: GPUs to train on (1 by default; the scheduler is set for 8).
- `MULTIDATASET_SER_01_WORKERS`: worker processes for `collect_stats`.
- `RESUME_CKPT`: a checkpoint (e.g. `exp/train_ecapa_tdnn/last.ckpt`) to
  resume training from.
- `MULTIDATASET_SER_01_EXP_TAG`: the directory under `exp/` to write to
  (`train_ecapa_tdnn` by default); give the same value to `infer measure`.
