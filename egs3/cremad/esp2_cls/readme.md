# CREMA-D speech emotion recognition recipe

Classifies each utterance into one of six emotions (`angry`, `disgust`, `fear`,
`happy`, `neutral`, `sad`) with a frozen
[WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) frontend,
a Transformer encoder and a linear head.

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) is already
distributed as 16 kHz mono WAV, so nothing is converted: `create_dataset` only
writes one manifest per split. The labels come from the file names
(`<actor>_<sentence>_<emotion>_<intensity>.wav`), not from an annotation file.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `CREMAD` — an existing corpus tree, or where to clone it to
- `CREMAD_OUTPUT` — where the manifests are written

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

## Notes

The label distribution is close to uniform — 1,271 clips for each emotion except
`neutral`, which has 1,087 — so WA and UA stay close together here. That is
unlike MELD, where `neutral` dominates and the two diverge.

All 91 actors read the same 12 sentences, so lexical content carries no
information about the label. A model cannot shortcut through the words, which
makes this corpus a cleaner test of paralinguistic modelling than corpora whose
scripts differ by emotion.
