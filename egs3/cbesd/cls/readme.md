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

After the cleaning described below, 4,164 recordings are used: 2,096 English and
2,068 Telugu, 3.00 hours in total, 0.94 to 5.03 seconds long (median about
2.6 s). They are 48 kHz stereo as distributed and are converted to 16 kHz
mono. Classes: angry 694, disgust 699, fear 700, happy 693, neutral 684,
sad 694.

Things to know when reading the scores:

- The Kaggle text says 4,320 utterances (and 4,200 elsewhere); the release holds
  4,204 wav files. The formula it gives (35 children x 2 sexes x 5 sentences x
  6 emotions x 2 languages) works out to 4,200.
- File names read `<number>.<T|E><F|M>_<age>_<emotion>_<take>.wav` (for example
  `1.TF_12_angry_1.wav`). The Kaggle description says the first number is the
  child's number and the number after the sex letter is the child's age (6 to 12
  in the description; 6 to 13 in the data). Yet the same (sex, age, number)
  triple occurs in only some of the emotion folders, so the recordings of one
  child cannot be reliably put together. The split below is therefore a guess at
  "same child".
- Cleaning: 10 files sit in a folder that contradicts the emotion in their name
  (five `Angry` files in `FEAR` and five `fear` files in `ANGER`; each is a
  byte-identical copy of the correctly filed one) and are dropped. One
  `... - Copy.wav` file and two files whose identical audio appears under two
  different labels (`neutral` and `sad`) are dropped too.
- 25 more groups (52 files) hold byte-identical audio under one label. Only the
  first utterance id of each group is kept, so 27 files are dropped. In most of
  these groups the copies carry **different child numbers, and in five groups
  different ages** (for example `neutral` takes 1 to 5 appear both as child 50,
  age 8 and as child 70, age 7). The release gives no way to tell which child
  and age are right, so the child number and age of the kept file may be wrong.
  The emotion label is the same within every group. The groups were found by
  checksum.
- File names are inconsistent (`Angry`/`angry`/`anger`, `Disguist`, `neutal`,
  missing separators), so the emotion in the name is only compared with the
  folder by its first two letters.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset C-BESD (Children's Bilingual Emotion Speech Dataset; `BESD/<TELUGU|ENGLISH>/<EMOTION>/`, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 4,204 | 70 (per the description; see below) | 3.03 |

How these were counted: every `.wav` in the `BESD` folders (48 kHz stereo, 0.94 to 5.04 s; 2,100 Telugu and 2,104 English); hours from the file lengths. The 70 children are the number in the Kaggle description; the file names carry a child number, sex and age, but the same (sex, age, number) triple occurs in only some emotion folders, so the children cannot be counted reliably from the files.

### Files per label

| label (folder name in the release) | samples |
|---|---|
| ANGER | 699 |
| DISGUST | 700 |
| FEAR | 705 |
| HAPPY | 700 |
| NEUTRAL | 700 |
| SAD | 700 |

How these were counted: the emotion folder each file is in, summed over the `TELUGU` and `ENGLISH` folders.

### Issues in the release

- The description says 4,320 utterances and, elsewhere, 4,200; the formula it gives (35 children x 2 sexes x 5 sentences x 6 emotions x 2 languages) works out to 4,200; the release has 4,204. The description gives 2.5 hours; the files total 3.03 hours.
- 10 files sit in a folder that contradicts the emotion in their name (five `Angry` files in `FEAR` and five `fear` files in `ANGER`); each is a byte-identical copy of the correctly filed one. One `... - Copy.wav` file exists, and two files have identical audio under two labels (`neutral` and `sad`).
- 25 groups (52 files) hold byte-identical audio under one label: 2 groups share the child and emotion and differ only in the utterance number, 23 carry different child numbers (in 5 of them also a different age). Which child number or age is correct cannot be told.
- Observation by the recipe author (2026-10-23): the 25 identical groups were listened to and are the same sound.
- Ages in the file names run from 6 to 13 although the description says 6 to 12.


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
| train | 2,913 | 2,913 |
| valid | 419 | 419 |
| test | 832 | 832 (not filtered) |

Each split is about half Telugu and half English (train 1,446 Telugu / 1,467
English, valid 209 / 210, test 413 / 419), and the six labels are nearly
balanced within it (test: 134 to 140).

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
