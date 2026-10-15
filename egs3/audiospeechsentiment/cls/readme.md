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
