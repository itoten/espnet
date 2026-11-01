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

The Kaggle description says the author prepared lines for each emotion and a native speaker
read them in that emotion's tone, so the sentence meaning differs by class; it does not say
how the emotion was expressed in the voice. See "Issues in the release" under "The data".

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SARAIKI_SER` — the directory holding the extracted `Angry/`, `Fear/`, `Happy/`, `Neutral/`, `Sad/` and `Shock/` folders
- `SARAIKI_SER_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset `mrjunaidsiyal/speech-emotion-recognition-saraiki-audio-dataset` (downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,200 | unknown | 1.08 |

How these were counted: every audio file in the six emotion folders (opus, mono, encoded at 48 kHz; 1.08 to 7.52 s, mean 3.2 s); hours from the file lengths. The description speaks of "a native Seraiki speaker" in the singular, and the distribution of F0 has a single peak, but the number of speakers is not stated, so it is unknown.

### Files per label

| label (folder name in the release) | samples |
|---|---|
| Angry | 200 |
| Fear | 200 |
| Happy | 200 |
| Neutral | 200 |
| Sad | 200 |
| Shock | 200 |

How these were counted: the emotion folder each file is in.

### Issues in the release

- 5 pairs of consecutive files in the same folder (for example `Neutral_023` and `Neutral_024`) are byte-identical.
- The description gives about 4 to 5 s per file; the files are 1.08 to 7.52 s (mean 3.2 s).
- The description says the author prepared lines for each emotion and a native speaker read them in that emotion's tone; the sentence meaning differs by class. It does not say how the emotion was expressed in the voice.
- Observation by the recipe author (2026-10-25 at the latest): in a listening check, the speaker seemed to read the sentences without acting the emotion in the voice (a subjective impression of a few clips).


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
