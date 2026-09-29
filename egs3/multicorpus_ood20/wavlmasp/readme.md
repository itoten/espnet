# Multi-corpus speech emotion recognition, out-of-domain evaluation (ood20)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling + a
fully connected head) on ten corpora used in full, and scores every training
epoch on corpora it never sees.

| Role | Corpora | Utterances |
|---|---|---:|
| Training (every split of each corpus) | CREMA-D, EMNS, EmoDB, Emozionalmente, eNTERFACE'05, JL-Corpus, MELD, Quechua Collao, SUBESCO, Thorsten | 55,524 (55,086 after the 0.5–30 s filter) |
| Per-epoch evaluation (every split of each corpus) | EmoV-DB, IEMOCAP, TESS, SAVEE, RAVDESS | 17,184 |
| Per-epoch evaluation | MSP-Podcast Development | 25,729 |
| Final test (chosen epochs only) | MSP-Podcast Test1 / Test2 | 35,728 / 11,363 |

Each corpus is read through its own `egs3/<corpus>/wavlmasp` recipe, from the
unfiltered `manifest/` that recipe writes. Utterance counts are those
manifests' row counts. The MSP-Podcast recipe's manifests carry seven labels:
angry, disgust, fear, happy, neutral, sad and surprise.

All 20 labels found in the training corpora are kept as classes
(`dataset/config.yaml`'s `label_map`). Every evaluation label is one of them.

## How this recipe differs from the other multi-corpus recipes

- **No validation during training.** `trainer.limit_val_batches: 0`. The
  `valid` split is a 64-row placeholder taken from the training manifest,
  present only because `DataOrganizer` requires one.
- **Fixed length, every epoch kept.** 30 epochs without early stopping; an
  extra `ModelCheckpoint` writes `exp/<tag>/epochs/epochNNN.ckpt`
  (weights only, 0-based `NNN`), about 1.27 GB each.
- **Model selection after training.** Each saved epoch is scored with
  `conf/inference.yaml`; the epoch to report is chosen afterwards from those
  scores, so any evaluation set can play the validation role without
  retraining.
- `src/inference.py` loads either a plain state dict or a Lightning
  checkpoint, so `epochs/*.ckpt` are used as-is.

## Running

Run each line as its own command. Under multi-GPU DDP, Lightning re-runs the
whole command once per GPU, so combining data preparation or inference with
`train` repeats them.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels collect_stats \
    --training_config conf/training.yaml
python run.py --stages train --training_config conf/training.yaml
```

Score one epoch (`INFERENCE_EPOCH` is the zero-padded 0-based index; the
config fails to resolve without it). Results go to
`exp/train_wavlmasp_ood20/inference/epochNNN/`:

```bash
INFERENCE_EPOCH=007 python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`infer` runs on one GPU per process. To use several GPUs, start one process
per epoch with a different `CUDA_VISIBLE_DEVICES` each; different epochs write
to different directories.

For the final test, use `conf/inference_test.yaml` the same way; its results
land in the same `epochNNN/` directory.

To resume an interrupted training run:

```bash
RESUME_CKPT=exp/train_wavlmasp_ood20/last.ckpt \
    python run.py --stages train --training_config conf/training.yaml
```

## Scoring

`conf/metrics.yaml` reuses the five `wavlmasp` metrics. The model predicts
over all 20 classes; UA and MacroF1 average over the classes that appear in
each evaluation set's references.
