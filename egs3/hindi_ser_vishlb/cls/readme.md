# Hindi SER (vishlb) speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Hindi SER (vishlb) alone and scores it on its own test split, classifying
each utterance into one of eight classes.

[Hindi SER](https://www.kaggle.com/datasets/vishlb/speech-emotion-recognition-hindi)
is a Kaggle dataset of acted Hindi speech: 8 speakers each read the same 10 sentences
in 8 classes over 5 sessions, 3,200 wav files in total (400 per class). The classes
are `angry` (folder `anger`), `disgust`, `fear`, `happy`, `neutral`, `sad`,
`sarcastic` and `surprise`; `sarcastic` is kept as released. Speaker, session and
label come from the folders (`<speaker>/session<N>/<emotion>/`), because two file
names carry typos (`dusgust`, `happpy`). In a sample of 160 source files, 140 were
16 kHz and 20 were 48 kHz, all mono, so `create_dataset` converts every file to
16 kHz with ffmpeg. The clips are 1.15 to 6.1 seconds long and 2.85 hours in total.

Environment variables (both optional; defaults are `download/` and `data/` under the
recipe):

- `HINDI_SER_VISHLB` — the directory holding the extracted `my Dataset/` folder
- `HINDI_SER_VISHLB_OUTPUT` — where the converted audio and manifests are written

## Download

The Kaggle zip (0.39 GB) needs a login, so the recipe does not download it. Download
it by hand, extract it and point `$HINDI_SER_VISHLB` at the folder holding
`my Dataset/`; otherwise `create_dataset` stops with the URL. On one run
`create_dataset` took 146 seconds. The download time was not measured.

## Splits

No partition is published. All speakers read the same 10 sentences, so whole speakers
are held out (`n_test_groups`, `n_valid_groups`, `split_seed` in
`dataset/config.yaml`): speakers 2 and 5 for test, 6 for valid and 1, 3, 4, 7 and 8
for training. Scores are therefore speaker-independent, but test sentences also
appear in training. Every class has the same count in each split.

| split | speakers | utterances | hours | after `remove_long_short` | per class |
|---|---|---|---|---|---|
| train | 1, 3, 4, 7, 8 | 2,000 | 1.79 | 2,000 | 250 |
| valid | 6 | 400 | 0.43 | 400 | 50 |
| test | 2, 5 | 800 | 0.63 | 800 (not filtered) | 100 |

## License

The Kaggle licence field reads "Database: Open Database, Contents: (c) Original
Authors". The terms of the contents have not been checked further.

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
