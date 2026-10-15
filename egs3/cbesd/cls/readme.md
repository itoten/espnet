# C-BESD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on C-BESD alone and scores it on C-BESD's own test split, classifying each
utterance into one of six emotions.

C-BESD (Children's Bilingual Emotion Speech Dataset) is an acted corpus of
children reading short sentences in Telugu and in English. The Kaggle
description gives 70 gender-balanced children aged 6 to 12 whose first language
is Telugu, recorded through the SurveyLex platform, with 6 emotions (`angry`,
`disgust`, `fear`, `happy`, `neutral`, `sad`) and 5 takes each. The copy used
here has `BESD/<TELUGU|ENGLISH>/<EMOTION>/` folders with 4,204 wav files; the
folder gives the label.

After the cleaning described below, 4,191 recordings are used: 2,097 English and
2,094 Telugu, 3.02 hours in total, 0.94 to 5.03 seconds long (median about
2.6 s). They are 48 kHz stereo as distributed and are converted to 16 kHz
mono. Classes: angry 694, disgust 700, fear 700, happy 699, neutral 699,
sad 699.

Things to know when reading the scores:

- The Kaggle text says 4,320 utterances (and 4,200 elsewhere); the release holds
  4,204 wav files. The shortfall is not explained.
- No child id is published. File names read `<number>.<T|E><F|M>_<n>_<emotion>_<take>.wav`
  (for example `1.TF_12_angry_1.wav`); `<n>` ranges from 6 to 13 in the data and
  is presumably the age, but this is not documented. The running number
  restarts in every emotion folder, and the same (sex, `<n>`, number) triple
  occurs in only some of the emotion folders, so the recordings of one child
  cannot be reliably put together. The split below is therefore a guess at
  "same child".
- Cleaning: 10 files sit in a folder that contradicts the emotion in their name
  (five `Angry` files in `FEAR` and five `fear` files in `ANGER`; each is a
  byte-identical copy of the correctly filed one) and are dropped. One
  `... - Copy.wav` file and two files whose identical audio appears under two
  different labels (`neutral` and `sad`) are dropped too. 26 other groups of
  byte-identical files with a single label remain.
- File names are inconsistent (`Angry`/`angry`/`anger`, `Disguist`, `neutal`,
  missing separators), so the emotion in the name is only compared with the
  folder by its first two letters.

## Provenance

Kaggle dataset `pranuthi19/bilingual-emotion-speech-datasetbesd` (version 1,
2024-02-07), licence shown on Kaggle: CC BY-SA 4.0. The required citation was
not checked; see the Kaggle page.

Kaggle requires a login, so the recipe does not download. Download the zip and
extract it into `$CBESD` so that `BESD/` lies inside it.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `CBESD` — the directory holding `BESD/`
- `CBESD_OUTPUT` — where the converted audio and manifests are written

## Download

The zip (1.04 GB) took 37 seconds to download. `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 4 minutes 52
seconds on the first run (the audio conversion dominates).

## Splits

No partition is published. Recordings are grouped by language, sex, the number
after the sex letter and the running number; groups sharing byte-identical audio
are merged. Whole groups go to one split, largest first, each to the split
furthest below its 70/10/20 target for the groups' language and labels
(`split_seed` in `dataset/config.yaml`). Children are not guaranteed to be held
out, because the grouping is a guess.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 2,931 | 2,931 |
| valid | 420 | 420 |
| test | 840 | 840 (not filtered) |

Each split is about half Telugu and half English (train 1,464 Telugu / 1,467
English, valid and test exactly half), and the six labels are balanced within it
(test: 140 each).

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
