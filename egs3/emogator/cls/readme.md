# EmoGator vocal-burst emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoGator alone and scores it on EmoGator's own test split, classifying each
recording into one of 30 emotion categories.

[EmoGator](https://github.com/fredbuhl/EmoGator) ([arXiv 2301.00508](https://arxiv.org/abs/2301.00508))
is a corpus of non-speech vocal bursts (for example laughter, sighs and groans), so
there is no linguistic content and no language. The 357 contributors (volunteers and
crowd workers, per the paper) were each given text prompts describing a situation for
every category and recorded three instances per category, which gives 90 recordings per
contributor. Measured on the downloaded files: 32,130 mp3 files, 16.77 hours in total
(the paper reports 16.97 hours), 0.42 to 2.04 seconds long, exactly 1,071 per
category. Sampling rates are 48 kHz (32,040 files) and 44.1 kHz (90 files); 30,424
files are mono and 1,706 stereo. Twenty pairs of files from the same contributor and
category are byte-identical copies; they are kept.

The label is the category number in the file name `<contributor>-<category>-<instance>.mp3`;
`dataset/config.yaml` maps `01`-`30` to the labels (the repository README order):
`adoration`, `amusement`, `angry`, `awe`, `confusion`, `contempt`, `contentment`,
`desire`, `disappointment`, `disgust`, `distress`, `ecstasy`, `elation`,
`embarrassment`, `fear`, `guilt`, `interest`, `neutral`, `pain`, `pride`,
`realization`, `relief`, `romantic_love`, `sad`, `serenity`, `shame`,
`surprise_negative`, `surprise_positive`, `sympathy`, `triumph`. `Anger` and `Sadness`
are written `angry` and `sad` to match the other recipes. `create_dataset` converts
the recordings to 16 kHz mono wav with ffmpeg.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the GitHub repository `fredbuhl/EmoGator` (mp3 files; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from GitHub, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 32,130 | 357 | 16.77 |

How these were counted: every `.mp3` in the repository (0.42 to 2.04 s; 48 kHz in 32,040 files and 44.1 kHz in 90; 30,424 mono and 1,706 stereo); hours from the file lengths. Speakers are the 357 contributor numbers in the file names `<contributor>-<category>-<instance>.mp3`. The release has no gender, age or language information.

### Files per label

| label (category number in the file name; names from the repository README) | samples | hours |
|---|---|---|
| 01 adoration | 1,071 | 0.557 |
| 02 amusement | 1,071 | 0.556 |
| 03 anger | 1,071 | 0.555 |
| 04 awe | 1,071 | 0.560 |
| 05 confusion | 1,071 | 0.557 |
| 06 contempt | 1,071 | 0.557 |
| 07 contentment | 1,071 | 0.562 |
| 08 desire | 1,071 | 0.562 |
| 09 disappointment | 1,071 | 0.560 |
| 10 disgust | 1,071 | 0.554 |
| 11 distress | 1,071 | 0.558 |
| 12 ecstasy | 1,071 | 0.565 |
| 13 elation | 1,071 | 0.559 |
| 14 embarrassment | 1,071 | 0.559 |
| 15 fear | 1,071 | 0.556 |
| 16 guilt | 1,071 | 0.560 |
| 17 interest | 1,071 | 0.559 |
| 18 neutral | 1,071 | 0.560 |
| 19 pain | 1,071 | 0.559 |
| 20 pride | 1,071 | 0.560 |
| 21 realization | 1,071 | 0.560 |
| 22 relief | 1,071 | 0.560 |
| 23 romantic love | 1,071 | 0.561 |
| 24 sadness | 1,071 | 0.561 |
| 25 serenity | 1,071 | 0.562 |
| 26 shame | 1,071 | 0.559 |
| 27 surprise (negative) | 1,071 | 0.553 |
| 28 surprise (positive) | 1,071 | 0.556 |
| 29 sympathy | 1,071 | 0.559 |
| 30 triumph | 1,071 | 0.560 |

How these were counted: the category number in the file name; every category has exactly 1,071 files (357 contributors x 3 instances). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The paper reports 16.97 hours; the files total 16.77 hours.
- 20 pairs of files from the same contributor and category are byte-identical.
- The category is the prompt given to the contributor; the release has no listener rating. The authors write that some categories can sound alike (for example crying of sadness and laughter).
- Quality control was done by one of the paper's authors only (silence, noise and reading out the category name were checked and re-recordings requested); the number of contributors excluded because the emotion did not come across is not given.


## How the labels were made

(From the paper, Buhl, [arXiv 2301.00508](https://arxiv.org/abs/2301.00508); read on 2026-10-24.)

- **The label is what the speaker meant to express, not what a listener heard.**
  Each contributor was shown a text prompt describing a situation for each of the 30
  categories and recorded three bursts per category; the category of the prompt is
  the label. The paper contrasts this with Hume-VB, whose samples carry ratings from
  listeners, which "may in some cases be very different than the speaker's intent".
  There is no listener evaluation in EmoGator. The paper itself suggests the corpus
  could be used to measure how often listeners recognise the intended category, and
  notes that some categories (for instance cries of sadness and of amusement) can
  sound alike.
- **Quality control was done by one person.** The paper's author was the sole
  evaluator. Some submissions were silent, contained only noise or
  read the category names aloud; contributors whose samples had background noise or
  breath noise were asked to re-record them, and contributors whose responses
  "didn't convey distinct emotions" were left out of the dataset. How
  many were left out is not stated.
- **Possible bias.** The paper says the dataset almost certainly leans towards the
  emotional expressions of North American English-speaking people, because the sole
  evaluator shares that background. Contributors' sex, age, language and origin are
  not in the release.
- The paper says efforts were made to keep one person from contributing twice, but
  this cannot be guaranteed.

## Provenance

GitHub repository `fredbuhl/EmoGator`, pinned to commit
`51eeca515a95f168ab73391cd9e7c975ab964429` (the latest at the time of writing). The
repository's licence is Apache-2.0. The recipe downloads the commit's tarball
(about 490 MB), which has no published checksum, so the download is checked only by
counting the mp3 files.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOGATOR` — a directory holding the `data/mp3/` files (anywhere below it), or where to download them to
- `EMOGATOR_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the tarball and extracts the mp3 files. Measured on one
run: about 30 seconds to download. `create_dataset` (mp3 to 16 kHz wav, 8 threads)
took about 203 seconds, `remove_long_short` 18 seconds and `prepare_labels` under 1
second.

## Splits

The corpus publishes no partition. Whole contributors are held out
(`n_test_speakers`, `n_valid_speakers`, `split_seed` in `dataset/config.yaml`): with
seed 0, 71 contributors (20%) form the test set, 36 (10%) the valid set and the
remaining 250 the training set. Every contributor has all 30 categories three times,
so every category has the same count within a split: 750 recordings per category in
train, 108 in valid and 213 in test.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 22,500 | 22,500 |
| valid | 3,240 | 3,240 |
| test | 6,390 | 6,390 (not filtered) |

`remove_long_short` drops nothing.

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

**Duplicates.** 20 recording(s) are byte-identical to another one (one per group is kept). They are listed in `duplicate_ids` and are dropped only when `exclude_duplicates` in `dataset/config.yaml` is `true` (default `false`).
