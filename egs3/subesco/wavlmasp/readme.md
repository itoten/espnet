# subesco speech emotion recognition (single-corpus baseline)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on subesco alone, and scores it on subesco's own test split (7
classes: `angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`/`surprise`).

[SUBESCO](https://doi.org/10.5281/zenodo.4526477) is 7,000 Bangla utterances:
20 actors (10 male, 10 female) each performing ten sentences in seven
emotions with five takes apiece. Every field comes from the file name
(`F_01_OISHI_S_1_ANGRY_1.wav`). The audio is 48 kHz, so `create_dataset`
resamples it to 16 kHz. `SUBESCO` points at an existing corpus tree, or where
to download it to; the Zenodo archive is checked against its published MD5.

No partition is published; speakers are dealt out in numeric order, the same
number of each gender per split, so the split is balanced by construction.

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

`SUBESCO_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
