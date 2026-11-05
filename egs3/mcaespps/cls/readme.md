# MCAE-SPPS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MCAE-SPPS alone and scores it on MCAE-SPPS's own test split, classifying each
utterance into one of seven emotions.

[MCAE-SPPS](https://osf.io/9jyzc/) (Mandarin Chinese Auditory Emotion database
of Subject-Personal-Pronoun Sentences, version 2025.11.30) is acted Mandarin
Chinese speech: 6 actors read sentences in `neutral` and the six basic emotions
`angry`, `disgust`, `fear`, `happy`, `sad` and `surprise`. The release holds 6,675
validated recordings, 2.30 hours in total, each 0.38 to 3.98 seconds long. The
labels are the seven shared by the other recipes.

The label comes from the file name, `<speaker><emotion><sentence>.wav` (e.g.
`33189.wav` is speaker 3, emotion 3 = anger, sentence 189). Each file also sits
in a folder named after its emotion, and `create_dataset` fails if the two
disagree. The recordings are 48 kHz mono, so `create_dataset` resamples them to
16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MCAESPPS` — the directory holding the extracted emotion folders, or where to download them to
- `MCAESPPS_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the OSF project `9jyzc` (MCAE-SPPS, version 2025.11.30; the emotion folders of audio; downloaded around 2026-10-25). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from OSF, no sign-in. The recordings that were not validated are not distributed.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 6,675 | 6 | 2.30 |

How these were counted: every `.wav` in the seven emotion folders (48 kHz mono 16-bit, 0.38 to 3.98 s); hours from the file lengths. Speakers are the 6 actors (speakers 1 to 3 male and 4 to 6 female, per the file names `<speaker><emotion><sentence>.wav`). The grid of 6 actors x 7 emotions x 200 sentences would have 8,400 recordings; 6,675 are distributed.

### Files per label

| label (emotion folder in the release) | samples | hours |
|---|---|---|
| neutral | 1,196 | 0.387 |
| sadness | 1,187 | 0.543 |
| anger | 1,129 | 0.337 |
| surprise | 969 | 0.308 |
| happiness | 785 | 0.245 |
| disgust | 738 | 0.246 |
| fear | 671 | 0.230 |

How these were counted: the emotion folder each file is in (the digit in the file name agrees). The emotion of the table (`Corpus information.xlsx`) after validation equals the folder for every file. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- Of the 8,400 recordings of the full grid, only the 6,675 selected by the validation are distributed.
- No duplicate recordings were found.

### Listening test ratings in the release

`Corpus information.xlsx` on the OSF project gives, for every recording, the result of a perception test: 720 raters in 18 groups, 40 raters per recording, chose the emotion and rated its intensity from 1 to 9. The share of the 40 who chose the intended emotion has mean 0.763 and minimum 0.475; 137 recordings are below 0.5. (This table was not part of the downloaded audio folders.)


## Download

`prepare_source` downloads the seven per-emotion zip files (640 MB) from
[OSF](https://osf.io/9jyzc/), checks each against the MD5 OSF lists, and
extracts the WAV files. The corpus is CC BY-NC 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so all three splits contain
all 6 speakers.

| split | utterances |
|---|---|
| train | 4672 (`remove_long_short` drops none) |
| valid | 668 |
| test | 1335 |

## Listening test

A perception test covers every recording: 720 raters in 18 groups, 40 raters per recording, chose the emotion and rated its intensity from 1 to 9. The per-recording accuracy (the share of the 40 who chose the intended emotion) is in `Corpus information.xlsx` on the OSF project: mean 0.763, minimum 0.475, and 137 recordings below 0.5. The ratings are not used here.

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
