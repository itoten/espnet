# Audio Speech Sentiment speech sentiment recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Audio Speech Sentiment alone and scores it on its own test split,
classifying each utterance into one of three polarity classes. The classes are
sentiment polarities (`negative`, `neutral`, `positive`), not emotion categories.

[Audio Speech Sentiment](https://www.kaggle.com/datasets/imsparsh/audio-speech-sentiment)
is a Kaggle upload that gives no language, speaker, recording or source
information. The release holds `TRAIN/` (250 wav files, listed with a class in
`TRAIN.csv`), `TEST/` (110 wav files, no labels) and mel-spectrogram images. This
recipe uses only the 250 labelled files: `negative` 87, `positive` 82 and `neutral` 81.
The audio is 44.1 kHz stereo, 1.70 to 3.94 seconds long (0.18 hours); 78 of the
250 files are stored as 64-bit float rather than 16-bit PCM.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset `imsparsh/audio-speech-sentiment` (downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 360 | unknown | 0.27 |

How these were counted: every `.wav` in `TRAIN/` (250) and `TEST/` (110) (44.1 kHz stereo, 1.70 to 3.94 s); hours from the file lengths (`TRAIN` alone: 0.18). The release gives no speaker information, so the number of speakers is unknown.

### Files per label

| label (the `Class` column of `TRAIN.csv`) | samples |
|---|---|
| Negative | 87 |
| Positive | 82 |
| Neutral | 81 |
| (no label; `TEST/`) | 110 |

How these were counted: the `Class` column of `TRAIN.csv`, which lists the 250 `TRAIN/` files; the 110 files of `TEST/` have no label.

### Issues in the release

- The Kaggle page gives no language, speaker, recording method, source, labelling procedure or paper.
- Audio format is mixed: 240 files are 16-bit PCM and 120 are 64-bit float (all 44.1 kHz stereo).
- The 145 recordings of the same sentences carry the same label in every case (Whisper large-v3 transcripts of the files, not checked by hand): 67 sentences, 0 disagreements.
- Whisper large-v3 identified English; the speaker count and language were estimated lightly (F0 distribution suggests one speaker) and are not confirmed.
- Observation by the recipe author (2026-10-20 at the latest): some files were listened to; whether the voice is acted to differ by label was not established.


## About the labels

The labels are not emotion categories, and they are not usable for categorical
emotion recognition. A light check (Whisper large-v3 transcripts of the 250 files,
not reviewed by a person) found that the spoken sentences are English and that the
label follows the meaning of the sentence: 67 sentences are recorded two or three
times (145 files) and every repetition has the same label ("You have given a great
response." is `positive`, "Your choice is very bad." is `negative`). If the label
comes from the words rather than the voice, the dataset does not teach
paralinguistic information and also cannot serve as a sentiment task for speech;
whether the voice was performed to match the polarity was not checked by
listening. The F0 medians of the files fall in a narrow range (128 to 148 Hz for
the 5th to 95th percentile), which suggests one speaker, and the Whisper language
detector is often unsure (English for 201 of 250 files), which may point to an
accented speaker. None of this is stated by the uploader. The labelling procedure
is unknown.

Three environment variables control where the data lives. The last two are optional
and default to `download/` and `data/` under the recipe.

- `AUDIOSPEECHSENTIMENT` — the directory holding `TRAIN.csv` and `TRAIN/`
- `AUDIOSPEECHSENTIMENT_OUTPUT` — where the converted audio and manifests are written

## Download

The data is obtained manually; the recipe raises an error with the URL when it is
missing. Download the zip from
[Kaggle](https://www.kaggle.com/datasets/imsparsh/audio-speech-sentiment) (a free
Kaggle account is needed), unzip it and set `$AUDIOSPEECHSENTIMENT` to the
directory that holds `TRAIN.csv`. Measured on one run: about 10 seconds to download
the 0.24 GB zip with the Kaggle REST API; `create_dataset`, `remove_long_short` and
`prepare_labels` together took about 39 seconds (four recipes were prepared at the
same time).

## Splits

The release publishes no partition of the labelled files and no speaker id, so
the 250 files are split 70/10/20 at random within each label (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`). Speakers
cannot be held out, so scores may be optimistic if speakers repeat.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 176 | 176 |
| valid | 25 | 25 |
| test | 49 | 49 (not filtered) |

## License

The Kaggle page lists the license as CC0 (public domain). The page gives no source
for the audio, so the uploader's right to license it is not established.

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
