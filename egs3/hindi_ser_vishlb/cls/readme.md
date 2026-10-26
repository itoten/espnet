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
names carry typos (`dusgust`, `happpy`). Of the 3,200 source files, 2,800 are 16 kHz
and 400 are 48 kHz, all mono, so `create_dataset` converts every file to 16 kHz with
ffmpeg. The clips are 1.15 to 6.1 seconds long and 2.85 hours in total.

The Kaggle page has no description. The archive holds two documents that serve as the
description: a Word file ("dataset sentences and details") with the instructions given
to the speakers (10 neutral sentences about students and studying, with Hindi
translations of the 8 emotions; it says up to 10 sessions, but the data holds 5) and
a spreadsheet of the 8 speakers (sex, age 22 to 24; speakers 1 to 4 are women and 5
to 8 men) collected at ABVV University by Vishal Bhardwaj. The spreadsheet also holds
personal details of the speakers, which are not reproduced here and not used. The
files record no consent.

One pair of files is byte-identical: speaker 2, `neutral`, sentence 03, in sessions 2
and 5 (both in the test split). Both are kept.

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
Authors". On Kaggle's licence list this is one of two options under "Open Data
Commons" (the other being "Database: Open Database, Contents: Database Contents").
It means two things:

- The **database** (its structure and its selection and arrangement) is under the
  Open Database License (ODbL 1.0): attribute public use, and offer an adapted
  database under the ODbL as well.
- The **contents**, here the recordings, remain the copyright of the original
  authors. The ODbL states that it does not cover rights in individual contents
  ("this License does not cover any rights (other than Database Rights or in
  contract) in individual Contents", section 2.4), and the label names no licence
  for the recordings. So the terms of use of the audio itself are not stated.

The uploader and the collector named in the spreadsheet are the same person, but the
page does not say what the "original authors" are or what they allow. **The terms of
use of the recordings are to be confirmed by asking the author** (not done yet); until
then treat them as unknown and do not redistribute the audio.

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
