# E-SERAVD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on E-SERAVD alone and scores it on E-SERAVD's own test split, classifying each
utterance into one of six emotions.

E-SERAVD (EEPIS Speech Emotion Recognition Audio Video Dataset) is an
Indonesian-language corpus of short clips cut from films
([Kaggle](https://www.kaggle.com/datasets/ahmadnafibudianto/e-seravd-speech-emotion-recognition-av-dataset)).
The Kaggle archive holds three releases; this recipe uses `E-SERAVD 1.2`, whose
`02_Audio-Only/<NN_Emotion>/<film>/NN-NN-<film>-T-<clip>.wav` files number 3,000,
500 per emotion, from 11 films. The label is the emotion code in the file name
(the second number, which also prefixes the folder): `angry`, `sad`, `neutral`,
`happy`, `disgust` and `fear`. According to the dataset's README, three human
raters scored each clip and a clip kept its main emotion only when the mean
probability was at least 0.67. The recordings are 44.1 kHz stereo, so
`create_dataset` converts them to 16 kHz mono with ffmpeg. After conversion the
clips are 0.83 to 7.5 seconds long and 2.0 hours in total.

The two older releases in the archive (`E-SERAVD` and `E-SERAVD 1.1`) hold the
same 1,200 four-emotion clips. Of 200 randomly chosen `1.1` clips, 115 (57.5%) are
byte-identical to a `1.2` clip, and 81 of those 115 carry a different emotion
label in `1.2`. The README does not say how the labels were reassigned. This
recipe takes the `1.2` labels as published.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset `ahmadnafibudianto/e-seravd-speech-emotion-recognition-av-dataset`, release `E-SERAVD 1.2`, audio-only part (downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token. The archive holds three releases (`E-SERAVD`, `E-SERAVD 1.1`, `E-SERAVD 1.2`); the counts below are those of `E-SERAVD 1.2`.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 3,000 | unknown | 2.0 |

How these were counted: every `.wav` under `E-SERAVD 1.2/.../02_Audio-Only/` (44.1 kHz stereo 32-bit float, 0.83 to 7.5 s, from 11 films); hours from the file lengths. Speakers: the clips are film dialogue and the release lists no speaker, so the number is unknown.

### Files per label

| label (folder name in the release) | samples | hours |
|---|---|---|
| 01_Angry | 500 | 0.322 |
| 02_Sad | 500 | 0.319 |
| 03_Neutral | 500 | 0.364 |
| 04_Happy | 500 | 0.328 |
| 05_Disgust | 500 | 0.328 |
| 06_Fear | 500 | 0.338 |

How these were counted: the emotion folder each file is in (the number in the file name agrees). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- Labels differ between releases: of 200 randomly chosen `E-SERAVD 1.1` clips, 115 are byte-identical to a `1.2` clip, and 81 of those 115 carry a different emotion label in `1.2` (for example none of the 19 `sad` clips of `1.1` that were checked is `sad` in `1.2`). The README does not say how the labels were reassigned.
- The README's vote counts that the Fleiss' kappa is computed from contain 42 votes for disgust and 908 for fear, while 500 clips each carry those labels in `1.2`, which needs at least 1,000 votes each. The procedure that added these classes is not described.
- 12 pairs (24 files) of clips from the same film have nearly the same sound under different emotions. Overlaps of different lengths were not examined.
- The Kaggle page states 16 kHz audio; the files are 44.1 kHz stereo 32-bit float. Fleiss' kappa is given as 0.6713 (README), 0.61 (paper abstract) and 0.6338.
- Observation by the recipe author (2026-10-22 at the latest): the 12 same-sound pairs were listened to and sound the same.
- The release does not state the rights to the film dialogue.


## Excluded clips

Twelve pairs of `DRNT` clips (24 clips, all in the training split) hold the same
audio, with a correlation of at least 0.97, under two different emotion labels
(for example one `sad` and one `fear`). Which label is right cannot be decided,
so both clips of every pair are listed in `exclude_files` in
`dataset/config.yaml`, and `create_dataset` stops if one of the names is not
found. Pairs whose two clips are cut with different lengths were not searched
for. Training goes from 2,003 to 1,979 utterances.

Environment variables (both optional; defaults are `download/` and `data/` under the
recipe):

- `ESERAVD` — the directory holding the extracted `E-SERAVD 1.2/` folder
- `ESERAVD_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the Kaggle dataset with the Kaggle CLI (`kaggle`, which
needs an API token from https://www.kaggle.com/settings/api in
`~/.kaggle/access_token` or `$KAGGLE_API_TOKEN`), as `egs3/jlcorpus/cls` does. The
zip is about 6.4 GB because it also holds the video and two older releases, so only
the `.wav` files of `E-SERAVD 1.2` are extracted into `$ESERAVD`
(`extract_prefix` and `extract_suffix` in `dataset/config.yaml`), and the zip is
deleted. A zip downloaded by hand also works: extract the same files and point
`$ESERAVD` at the folder holding `E-SERAVD 1.2/`. On one run `create_dataset` took 130
seconds. The download time was not measured.

## Splits

The archive publishes no partition or speaker id. Whole films are held out as the
closest stand-in for speaker independence (`n_test_groups`, `n_valid_groups`,
`split_seed` in `dataset/config.yaml`): three films for test, one for valid and
seven for training (the exclusion above does not change which films go where).
Scores are therefore film-independent, but the label balance of
the held-out films is uneven; the single valid film has only 7 `sad` clips.

| split | films | utterances | hours | after `remove_long_short` | angry | disgust | fear | happy | neutral | sad |
|---|---|---|---|---|---|---|---|---|---|---|
| train | AADC, DHB, DRNT, HA3, HSL, KKNP, TAOL | 1,979 | 1.35 | 1,979 | 383 | 319 | 300 | 319 | 378 | 280 |
| valid | D1990 | 181 | 0.12 | 181 | 49 | 33 | 23 | 21 | 48 | 7 |
| test | ADNJ, PKGL, SWDN | 816 | 0.51 | 816 (not filtered) | 64 | 146 | 172 | 158 | 71 | 205 |

## License

The Kaggle page lists the dataset as CC BY 4.0. The clips are cut from films; the
rights to the film audio itself have not been checked.

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
