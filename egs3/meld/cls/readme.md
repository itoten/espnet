# meld speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on meld alone, and scores it on meld's own test split (7 classes:
`angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`/`surprise`, spelled to match
the other corpora here; MELD's own labels are `anger`/`disgust`/`fear`/`joy`/
`neutral`/`sadness`/`surprise`).

[MELD](https://affective-meld.github.io/) ships audio as MP4; `create_dataset`
downloads the corpus if not already present, extracts it, converts every clip
to 16 kHz mono WAV, and writes one manifest per split. `MELD` points at an
existing MELD source tree, or where to download it to.

`remove_long_short` filters the train and valid splits to 0.1-30s; the test
split is left untouched so the reported scores cover the whole set.

## Quick start

```bash
# 1) Write the manifests
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`MELD_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
