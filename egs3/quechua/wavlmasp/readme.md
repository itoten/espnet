# quechua speech emotion recognition (single-corpus baseline)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on quechua alone, and scores it on quechua's own test split (9
classes: `angry`/`boredom`/`calm`/`excited`/`fear`/`happy`/`neutral`/`sad`/
`sleepy`, spelled to match the other corpora here; the corpus's own spelling
is `anger`/`sadness`).

The [Quechua Collao corpus](https://doi.org/10.6084/m9.figshare.20292516) is
12,420 recordings by six actors performing 230 script items in nine emotions.
Labels come from `Data/Data/Data.xlsx`, read as a zip of XML rather than
adding a spreadsheet dependency. `QUECHUA` points at an existing corpus tree,
or where to download it to; the figshare archive is checked against its
published MD5.

No partition is published; speakers are dealt out in order (a1-a4 train, a5
valid, a6 test) after the metadata repairs `dataset/config.yaml` documents
(two mis-spelled emotions, two mis-typed speaker ids, 42 mis-attributed
`happy` recordings).

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

`QUECHUA_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
