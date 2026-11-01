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

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the ShEMO (Sharif Emotional Speech Database) release of the GitHub project `mansourehk/ShEMO` (`female.zip` and `male.zip` via Git LFS; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Download through Git LFS from GitHub, no sign-in. The Dropbox link at the top of the README is broken.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 3,000 | 87 | 3.43 |

How these were counted: every `.wav` in the two zips (mono; 44.1 kHz in 2,869 files and 22.05 kHz in 131; 0.36 to 33.33 s); hours from the file lengths. Speakers are the 87 speaker numbers in the file names `<gender><speaker><emotion><index>.wav` (31 female, 56 male; numbers restart for each gender).

### Files per label

| emotion code in the file name | read as (README) | samples |
|---|---|---|
| A | anger | 1,059 |
| N | neutral | 1,028 |
| S | sadness | 449 |
| W | surprise | 225 |
| H | happiness | 201 |
| F | fear | 38 |

How these were counted: the emotion letter of each file name. The release has no disgust class. According to the README, twelve annotators labelled each utterance and the final label is the majority vote; the individual ratings are not distributed.

### Issues in the release

- `M51N09.wav` (neutral) has the same bytes as `M50A03.wav` (angry); in the transcription zip the two files have different sentences, and a Whisper transcript of both files matches the sentence of `M50A03`, so the neutral label of `M51N09.wav` does not fit the sound.
- The license is only the README's statement that the database is free for academic purposes; the release has no license file.


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

## Listening test

Twelve annotators labelled each utterance and the label is the majority vote (Cohen's kappa about 64%, "substantial"). The individual ratings are not distributed.

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

**Duplicates.** The audio of `M51N09.wav` (`neutral`) is a copy of `M50A03.wav` (`angry`):
the files are byte-identical, and the transcription of both matches the sentence of
`M50A03`, so the `neutral` label of `M51N09` does not match its sound. Both are used by
default. `exclude_duplicates: true` in `dataset/config.yaml` drops `M51N09`
(`duplicate_ids`), one recording fewer; the default is `false`.
