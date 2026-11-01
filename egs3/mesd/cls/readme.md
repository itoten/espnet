# MESD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MESD alone and scores it on MESD's own test split, classifying each
utterance into one of six emotions.

[MESD](https://data.mendeley.com/datasets/cy34mh68j9/5) (Mexican Emotional
Speech Database) is Mexican Spanish single-word speech from non-professional
actors (3 female, 2 male and 6 child voices, per the dataset description),
with words drawn from two word corpora
(A and B). This recipe uses the 862 human recordings in six emotions:
`disgust`, `fear`, `happy`, `sad` (144 each), `angry` and `neutral` (143 each;
the release has one child recording fewer than the expected 144 for each).

The label comes from the file name, `<Emotion>_<voice>_<word corpus>_<word>.wav`
(e.g. `Fear_C_B_oscuridad.wav`, voice `F`/`M`/`C`); no annotation file is read.
The file name does not identify the speaker, only the voice type. The release
also holds 288 naturalness-reduced copies of female corpus-B recordings
(`<Emotion>_F_B_L<level>_<word>.wav`, edited from the human recordings); these
are skipped. The recordings are 48 kHz mono 24-bit, so `create_dataset`
resamples them to 16 kHz with ffmpeg.

The utterances are single words, from 0.20 s to 2.54 s long. `remove_long_short`
drops the one valid utterance shorter than 0.2 s (0.199 s).

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MESD` — an extracted copy of the release, or where to download it to
- `MESD_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Mendeley Data record `cy34mh68j9` version 5 (MESD, Mexican Emotional Speech Database; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Mendeley Data, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,150 | unknown (see below) | 0.22 |

How these were counted: every `.wav` in the release (48 kHz mono 24-bit, 0.20 to 2.54 s; the utterances are single words); hours from the file lengths. 862 files are human recordings and 288 are naturalness-reduced copies of female corpus-B recordings (`<Emotion>_F_B_L<level>_<word>.wav`). The file names give only the voice type (`F` 576 files, `M` 288, `C` 286), not the speaker; the description speaks of 3 female, 2 male and 6 child voices and the paper of 4 adult men, 4 adult women and 8 children (mean age 9.9 years), so the number of speakers is unknown.

### Files per label

| label (emotion in the file name) | human recordings | naturalness-reduced copies |
|---|---|---|
| Anger | 143 | 48 |
| Disgust | 144 | 48 |
| Fear | 144 | 48 |
| Happiness | 144 | 48 |
| Neutral | 143 | 48 |
| Sadness | 144 | 48 |

How these were counted: the emotion field of each file name `<Emotion>_<voice>_<word corpus>_<word>.wav`. According to the paper the recordings were selected by a cubic SVM classifier, not by human listeners, from 1,152 recorded per voice type.

### Issues in the release

- The paper expects 864 human recordings (144 per emotion); the release has 862 (one child `Anger` and one child `Neutral` recording are missing).
- `Happiness_C_A_dentro` and `Happiness_C_A_lento_dentro` have the same sound; the word in the names differs, so one of the names is probably wrong (inferred).
- Each voice type has its own speakers per emotion, so speaker and emotion are tied (stated in the paper).


## Download

`prepare_source` downloads version 5 of the dataset as one zip (101 MB) from
[Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5). The corpus is
CC BY 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 604 |
| valid | 84 |
| test | 174 |

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

**Speaker and emotion are tied.** Within each voice type (female, male, child) each
emotion was recorded by one speaker (or, for children, one pair); for example all
female `fear` recordings come from one speaker and all female `happy` recordings from
another, so the speaker's voice reveals the emotion. The file names do not give the
speaker; this is known from the paper. This is recorded here as a fact and the data
are used as they are.

**Selection.** The recordings are not rated by listeners. For each voice type the
authors recorded 1,152 words and kept the 288 whose emotions a cubic SVM classifier
separated best (paper); a different algorithm could select different recordings.
Corpus A uses the same 24 words across emotions and voice types, corpus B uses a
different set of 24 words per emotion. There is no listening test.

**Duplicates.** `Happiness_C_A_dentro` and `Happiness_C_A_lento_dentro` (both
child, `happy`) are the same sound but their names give different words (`dentro`
and `lento_dentro`), so one of the names may be a mistake (an inference). They are kept
by default. `exclude_duplicates: true` in `dataset/config.yaml` drops the second one
(`duplicate_ids`), leaving 861; the default is `false`.
