# emns speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on emns alone, and scores it on emns's own test split (8 classes:
`angry`/`disgust`/`excited`/`happy`/`neutral`/`sad`/`sarcastic`/`surprise`).

[EMNS](https://www.openslr.org/136/) is a single-speaker narrative-storytelling
corpus published as 48 kHz WebM; `create_dataset` transcodes it to 16 kHz mono
WAV with ffmpeg. The label comes from the pipe-separated `metadata.csv`.
`EMNS` points at an existing corpus tree (`cleaned_webm/` and `metadata.csv`),
or where to download it to.

The corpus has one speaker, so no split can hold speakers out; the partition
is stratified by emotion instead (20% valid, 20% test), which
`dataset/config.yaml`'s ratios and seed control.

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

`EMNS_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
