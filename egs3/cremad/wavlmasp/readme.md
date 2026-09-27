# cremad speech emotion recognition (single-corpus baseline)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on cremad alone, and scores it on cremad's own test split (6
classes: `angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`).

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) is already
16 kHz mono WAV, so nothing is converted. The label comes from the file name
(`<actor>_<sentence>_<emotion>_<intensity>.wav`); no annotation file is read.
`CREMAD` points at an existing corpus tree, or where to clone it to.

The corpus has no official partition. Speaker ids are sorted and dealt out to
train/valid/test, which keeps the split reproducible from the corpus alone;
`dataset/config.yaml` holds the per-split speaker counts.

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

`CREMAD_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
