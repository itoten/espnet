# emozionalmente speech emotion recognition (single-corpus baseline)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on emozionalmente alone, and scores it on emozionalmente's own test
split (7 classes: `angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`/
`surprise`, spelled to match the other corpora here; the corpus's own
spelling is `anger`/`joy`/`neutrality`/`sadness`).

[Emozionalmente](https://doi.org/10.5281/zenodo.12616095) is 6,902 Italian
utterances crowdsourced from 431 non-professional actors, already 16 kHz
mono WAV. `EMOZIONALMENTE` points at an existing corpus tree, or where to
download it to; the Zenodo archive is checked against its published MD5.

The corpus publishes a speaker-independent train/dev/test split, stratified
by emotion, gender and age, and this recipe reads it from
`metadata/split/*.csv` rather than rebuilding it.

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

`EMOZIONALMENTE_OUTPUT` must already point at the prepared data;
`create_dataset` writes it there if it does not.
