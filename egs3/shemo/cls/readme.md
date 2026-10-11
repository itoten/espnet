# ShEMO speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ShEMO alone and scores it on ShEMO's own test split, classifying each
utterance into one of six emotions.

[ShEMO](https://github.com/mansourehk/ShEMO) (Sharif Emotional Speech Database)
is Persian semi-natural speech cut from online radio plays: 3,000 utterances
from 87 speakers (31 female, 56 male), 3.4 hours. Twelve annotators labelled
each utterance and the final label is the majority vote. The labels are
`angry`, `fear`, `happy`, `neutral`, `sad` and `surprise`; there is no
`disgust`. The classes are very uneven: `angry` and `neutral` have about a
thousand utterances each, `fear` only 38.

The label comes from the file name, `<gender><speaker><emotion><index>.wav`
(e.g. `F14A09.wav`: female speaker 14, anger, 9th utterance); the emotion
codes are A, F, H, N, S and W (surprise). Speaker numbers restart for each
gender, so the speaker id is the gender and the number together. The
recordings are mono at 44.1 or 22.05 kHz, so `create_dataset` converts them to
16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SHEMO` — the extracted `female.zip` and `male.zip`, or where to download them to
- `SHEMO_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `female.zip` (374 MB) and `male.zip` (497 MB) from
the authors' GitHub repository, pinned to its latest commit. The files are Git
LFS objects, so they are fetched from `media.githubusercontent.com`. The
README offers the corpus "for academic purposes free of charge"; the repository
carries no separate license file.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 2099 |
| valid | 300 |
| test | 601 |

## Quick start

Run the three commands separately. With several GPUs, `train` starts one
process per GPU by re-running its own command, so any other stage listed in the
same command would run once per GPU as well.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels \
    collect_stats --training_config conf/training.yaml
python run.py --stages train \
    --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
