# JL-Corpus speech emotion recognition recipe

Classifies each utterance into one of ten emotions with a frozen
[WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) frontend,
a Transformer encoder and a linear head.

[JL-Corpus](https://github.com/tli725/JL-Corpus) is New Zealand English read by
four actors, built with an even distribution of four long vowels so that formant
and glottal-source features can be compared across emotions. Five of its ten
emotions are the usual primary ones (`angry`, `excited`, `happy`, `neutral`,
`sad`); the other five are secondary (`anxious`, `apologetic`, `assertive`,
`concerned`, `encouraging`) and are the reason the corpus exists. Each emotion
holds exactly 240 utterances, 60 per speaker.

The labels come from the file names, which
`Raw JL corpus (unchecked and unannotated)/Format_Intro.txt` documents as
`(Gender)(speaker.ID)_(Emotion)_(Sentence.ID)(session.ID)`; no annotation file
is read. The audio is 44.1 kHz, so `create_dataset` resamples it to 16 kHz with
ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `JLCORPUS` — an existing corpus tree, or where to download it to
- `JLCORPUS_OUTPUT` — where the resampled audio and manifests are written

## Download

The GitHub repository carries only the supporting documents — the audio is too
large for it — so the corpus comes from
[Kaggle](https://www.kaggle.com/datasets/tli725/jl-corpus) (CC0, 1.3 GB) and an
API token is needed:

```bash
pip install kaggle
# then put a token from https://www.kaggle.com/settings/api in
# ~/.kaggle/access_token, or export it as $KAGGLE_API_TOKEN
```

`egs2/l3das22/enh1` fetches its corpus the same way. Note that the archive
unpacks two copies of the tree that differ only in case
(`Raw JL corpus …` and `raw jl corpus …`); the builder reads one of them, but
they collide on a case-insensitive filesystem.

## Splits

The corpus publishes no partition, and it has four speakers. One is held out
for test; validation is drawn from the remaining three rather than held out
whole, because a second held-out speaker would leave two for training.
`dataset/config.yaml` holds both settings.

| split | speakers | utterances |
|---|---|---|
| train | female1, female2, male1 | 1440 |
| valid | female1, female2, male1 | 360 |
| test | male2 | 600 |

Two things follow, and both limit what the numbers mean.

**The test split is one male speaker.** With 2 male and 2 female speakers,
holding out one of each would balance it but cut training from 1440 utterances
to 960 — the worse trade at this size. Nothing here measures how the model does
on unseen female speakers. Listing a second speaker in `test_speakers` changes
this.

**Validation is not a held-out condition.** It shares speakers with training,
and because every speaker reads the same 15 sentences in two sessions with two
repetitions, a validation utterance usually has near-duplicates in training. It
is usable for choosing an epoch and nothing else — do not report it. Measured
here it read 30 points above test.

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

## Results

Test scores with `emotion_set: all`, the default:

| WA | UA | macro F1 | mAP | AUC |
|---|---|---|---|---|
| 51.83 | 51.83 | 45.87 | 57.49 | 89.50 |

WA and UA are identical because every split holds the same number of utterances
per emotion.

Ten classes from three training speakers is the hardest setting of any recipe
here, and it shows. `neutral` and `angry` absorb the errors — both recall above
96 % at precisions of 51.7 and 32.6 — while `happy` (recall 8.3), `encouraging`
(3.3), `concerned` (15.0) and `sad` (16.7) are largely missed. Separating
primary from secondary is not the problem: 92.7 % of primary utterances are
predicted as some primary emotion. The confusion is inside each group.

Set `emotion_set: primary` for the five basic emotions alone, which is what to
use when lining these labels up against a corpus that has no secondary ones.

## Notes

Each recording ships with a `.txt` holding its prompt. The manifests do not
carry it — the classification task reads only audio and label — but it is there
for work that needs the text.
