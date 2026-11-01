# RAVDESS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on RAVDESS alone and scores it on RAVDESS's own test split, classifying
each utterance into one of eight emotions.

[RAVDESS](https://zenodo.org/records/1188976) is English read speech from 24
actors (odd-numbered male, even-numbered female). This recipe
uses the speech, audio-only release: 1440 utterances in eight emotions,
`neutral`, `calm`, `happy`, `sad`, `angry`, `fear`, `disgust`, `surprise`.
`neutral` has half as many utterances as the others because it was recorded
at one intensity only.

The label comes from the file name,
`<modality>-<vocal channel>-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav`,
all two-digit codes; no annotation file is read. The recordings are 48 kHz and
five of them are stereo, so `create_dataset` converts all of them to 16 kHz
mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `RAVDESS` — an existing `Actor_NN/` tree, or where to download it to
- `RAVDESS_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the RAVDESS Zenodo record 1188976, the audio-only speech file `Audio_Speech_Actors_01-24.zip` (208 MB; downloaded around 2026-10-15). The record also holds the song, video and audio-video parts, which were not obtained. All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,440 | 24 | 1.48 |

How these were counted: every `.wav` under the `Actor_NN` folders (48 kHz; 1,435 mono and 5 stereo; 2.94 to 5.27 s); hours from the file lengths. Speakers are the 24 actors (odd numbers male, even numbers female).

### Files per label

| emotion code in the file name | read as (record description) | samples |
|---|---|---|
| 01 | neutral | 96 |
| 02 | calm | 192 |
| 03 | happy | 192 |
| 04 | sad | 192 |
| 05 | angry | 192 |
| 06 | fearful | 192 |
| 07 | disgust | 192 |
| 08 | surprised | 192 |

How these were counted: the third field of the file name `<modality>-<channel>-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav`. Neutral has half as many files because it was recorded at one intensity only.

### Issues in the release

- `03-01-03-01-02-01-07` and `03-01-03-01-02-02-07` (both happy; repetitions 1 and 2 of one actor, statement and intensity) have byte-identical content.


## Download

`prepare_source` downloads `Audio_Speech_Actors_01-24.zip` (208 MB) from
[Zenodo](https://zenodo.org/records/1188976). The corpus is CC BY-NC-SA 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Actors are not held out, so all three splits contain
all 24 actors.

| split | utterances |
|---|---|
| train | 1012 |
| valid | 143 |
| test | 285 |

## Listening test

**RAVDESS was validated by listeners** (Livingstone and Russo, 2018, PLOS ONE 13(5)
e0196391). All 7,356 recordings of the full database (speech and song, in audio-video,
video-only and audio-only form; this recipe uses the 1,440 audio-only speech files) were
rated 10 times each for emotional validity (the rater chose the emotion), intensity and
genuineness, by 247 untrained research participants from North America, undergraduate
students at Ryerson University; another 72 participants provided test-retest data. The overall
proportion of correct answers was 0.72 for speech (unbiased hit rate 0.53); interrater
agreement was substantial for strong expressions (kappa 0.62) and moderate for normal ones
(0.53). Over both speech and song, the audio-only presentations were recognised least
well: 0.60 correct (95% CI 0.59 to 0.61), against 0.81 for audio-video and 0.75 for
video-only. Per-recording scores are in the paper's supplementary table S1 and not in the
audio download; the recipe uses the intended emotion of the file name.

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

**Duplicates.** 1 recording(s) are byte-identical to another one (one per group is kept). They are listed in `duplicate_ids` and are dropped only when `exclude_duplicates` in `dataset/config.yaml` is `true` (default `false`).
