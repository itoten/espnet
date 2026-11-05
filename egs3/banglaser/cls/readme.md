# BanglaSER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BanglaSER alone and scores it on BanglaSER's own test split, classifying each
utterance into one of five emotions.

[BanglaSER](https://data.mendeley.com/datasets/t9h6p943xy/5) (Das, Islam and
Ahmed; Data in Brief 42, 2022) is acted Bangla speech from 34 non-professional
speakers (17 male, 17 female, aged 19 to 47), each reading 3 statements 3 times
in `angry`, `happy`, `sad` and `surprise`, and 27 of them also in `neutral`.
There are 1,467 recordings (306 for each of the four emotions, 243 `neutral`),
1.37 hours in total, each 2.6 to 4.0 seconds long. There is no `disgust` or `fear`.

The label comes from the file name, in the RAVDESS style,
`<mode>-<type>-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav`
(e.g. `03-01-03-02-02-02-05.wav`); the emotion codes are 01 happy, 02 sad,
03 angry, 04 surprise and 05 neutral, as in the paper's Table 4. Two quirks in
the release are handled: the three surprise recordings of actor 15 have mode
`01` instead of `03`, and actor 19's folder holds a file named like actor 3's
(`03-01-05-01-01-03-03.wav`, with different audio). The speaker is therefore
taken from the `Actor NN` folder, and utterance ids include it. The recordings
are 44.1 or 48 kHz, mono or stereo; `create_dataset` converts them to 16 kHz mono
with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BANGLASER` — the directory holding the extracted audio, or where to download it to
- `BANGLASER_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Mendeley Data record `t9h6p943xy` version 5 (BanglaSER; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Mendeley Data, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,467 | 34 | 1.37 |

How these were counted: every `.wav` in the archive (44.1 or 48 kHz, mono or stereo; 2.62 to 3.99 s); hours from the file lengths. Speakers are the 34 `Actor NN` folders (17 male, 17 female; odd numbers male and even numbers female, per the record).

### Files per label

| emotion code in the file name | read as (paper's Table 4) | samples | hours |
|---|---|---|---|
| 01 | happy | 306 | 0.285 |
| 02 | sad | 306 | 0.288 |
| 03 | angry | 306 | 0.283 |
| 04 | surprise | 306 | 0.290 |
| 05 | neutral | 243 | 0.228 |

How these were counted: the third field of the RAVDESS-style file name `<mode>-<type>-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav`. Neutral exists for 27 speakers only (Actors 11, 12, 13, 16, 18, 20 and 29 have none). The release has no disgust or fear class. The intensity field is determined by the emotion (angry and surprise are strong, the others normal). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- 8 pairs (16 files) have the same sound with the same label: 7 are other repetitions of one speaker, statement and emotion (for example Actor 15, happy, statement 3, repetitions 1 and 3) and 1 is surprise statement 3 repetition 1 of Actors 16 and 19.
- The three surprise files of Actor 15 have mode `01` instead of `03`; the folder of Actor 19 holds a file named like one of Actor 3 (`03-01-05-01-01-03-03.wav`) with different audio.
- The paper gives 1 hour 29 minutes in total and 44.1 kHz; the files total 1.37 hours and mix 44.1 and 48 kHz, mono and stereo.


## Download

`prepare_source` downloads the archive that Mendeley Data builds on request
(446 MB) and extracts the WAV files. The corpus is CC BY 4.0. Mendeley publishes no
checksum for it, so completeness is checked by counting the 1,467 files.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share all
34 speakers.

| split | utterances |
|---|---|
| train | 1022 |
| valid | 144 |
| test | 293 |

## Listening test

The paper reports a perception test with 15 raters (8 men, 7 women; students and staff of two universities), about 33 clips each: the intended emotion was perceived in 80.5% of the cases (the most frequently chosen emotion counts). Surprise was the hardest (confused with happy). Only the confusion matrix is in the paper; the ratings are not distributed.

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

**Duplicates.** 8 pairs of recordings have the same sound (identical after decoding, or a waveform correlation of 0.998 or more), all with the same label. Seven are another repetition of the same speaker, sentence and emotion; one is the same `surprise` clip in the folders of actors 16 and 19. With `exclude_duplicates: true` (the default) one of each pair is dropped (`duplicate_ids`, 1,459 recordings left); `false` keeps both.

**Intensity labels.** The intensity field of the file name is not independent of the emotion: `angry` and `surprise` are always strong and the others always normal. Do not use it as a separate label, or as evidence about intensity, without accounting for this.
