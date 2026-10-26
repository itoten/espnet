# Saraiki speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on the Saraiki speech emotion dataset alone and scores it on its own test
split, classifying each utterance into one of six emotions.

[Saraiki Speech Emotion Recognition dataset](https://www.kaggle.com/datasets/mrjunaidsiyal/speech-emotion-recognition-saraiki-audio-dataset)
(Kaggle, CC BY 4.0) holds 1200 Saraiki-language acted utterances, 200 in each of
six folders: `angry`, `fear`, `happy`, `neutral`, `sad` and `shock`. The
recordings are 1.08 to 7.52 s long (1.08 hours in total), 48 kHz mono opus.
`create_dataset` converts them to 16 kHz wav with ffmpeg. The label is the folder name.
Five pairs of consecutive files in the same folder (for example
`Neutral_023` and `Neutral_024`) are byte-identical; they are kept as they are. No speaker id is published, and the
Kaggle page does not state the number of speakers.

**Warning: this corpus may carry no paralinguistic emotion. Handle it with great
care.** The Kaggle description says the author prepared lines for each emotion and
a native speaker read them in that emotion's tone, so the sentence meaning
differs by class. It does not say the speaker acted the emotion through voice or
prosody, and in a listening check a single speaker seemed simply to read the
sentences. If so, a model can score well by learning the words (or the recording
setup) alone, and the corpus may mislead training, especially when it is mixed
with corpora whose emotion is in the voice. Be very careful before using it in
multi-corpus training or reading its scores as speech emotion recognition; a
text-only baseline on a transcript would show how much the words alone explain.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SARAIKI_SER` — the directory holding the extracted `Angry/`, `Fear/`, `Happy/`, `Neutral/`, `Sad/` and `Shock/` folders
- `SARAIKI_SER_OUTPUT` — where the converted audio and manifests are written

## Download

The recipe does not download the data. Download the zip from the Kaggle page
above (8.4 MB, free Kaggle account) and unzip it into `$SARAIKI_SER`. If the folders
are missing, `create_dataset` stops with an error that gives the URL. Measured on
one run, the download took a few seconds (not timed) and `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 78 seconds.

## Splits

The dataset publishes no partition and no speaker id, so speakers cannot be
held out. Utterances are split 70/10/20 at random within each label
(`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`).
If the recordings come from few speakers, scores are optimistic.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 840 | 840 |
| valid | 120 | 120 |
| test | 240 | 240 (not filtered) |

Every split holds the same number of utterances per label.

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

**Duplicates.** 5 recording(s) are byte-identical to another one (one per group is kept). They are listed in `duplicate_ids` and are dropped only when `exclude_duplicates` in `dataset/config.yaml` is `true` (default `false`).
