# subesco speech emotion recognition (single-corpus baseline)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on subesco alone, and scores it on subesco's own test split.
Labels are subesco's native set (7 classes), the same
set `egs3/multicorpus_full20/wavlmasp` uses for subesco's test scoring, so
the two are directly comparable: same labels, same test data, the only
difference is whether training data came from subesco alone or from the
ten-corpus merge. See `espnet_work/per_corpus_baseline_plan.md`.

`max_epochs` (30) and `EarlyStopping.patience` (15) are
scaled up from the multicorpus recipes' defaults so the optimizer sees at
least ~3000 updates despite this corpus's smaller training set (see the
plan doc §4 for the derivation); the multicorpus defaults left EmoDB
under-trained at ~350 updates.

## Quick start

```bash
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml
python run.py --stages collect_stats train \
    --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`SUBESCO_OUTPUT` must already point at subesco's own prepared
data (same as `egs3/subesco/esp2_cls`); `SUBESCO_WAVLMASP_OUTPUT` controls where this
recipe's own filtered manifests and label list are written.
