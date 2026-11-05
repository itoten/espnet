# JVNV speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on JVNV alone and scores it on JVNV's own test split, classifying each
utterance into one of six emotions.

[JVNV](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus)
(Japanese emotional speech corpus with Verbal content and Nonverbal
Vocalizations; arXiv:2310.06072) is acted Japanese speech from 4 professional
speakers (F1, F2, M1, M2) in `angry`, `disgust`, `fear`, `happy`, `sad` and
`surprise`. Every utterance contains at least one nonverbal vocalization such as
laughter or sobbing. There are 1,615 recordings, 3.94 hours in total, each 3.2 to
19.8 seconds long. There is no `neutral`.

The label comes from the file name,
`<speaker>_<emotion>_<session>_<item>.wav` (e.g. `F1_anger_regular_01.wav`);
each file also sits in `<speaker>/<emotion>/<session>/`, and `create_dataset`
fails if the folders disagree with the name. The session is `regular`
(1,375 recordings; the nonverbal phrase was designated) or `free` (240; the
speaker chose it). The recordings are 48 kHz mono 24-bit; `create_dataset`
converts them to 16 kHz 16-bit with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `JVNV` — the directory holding the extracted audio, or where to download it to
- `JVNV_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the JVNV corpus (`jvnv_v1` from the corpus page; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from the corpus page, no sign-in. The download has no checksum; completeness was checked by the 1,615 file count.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,615 | 4 | 3.94 |

How these were counted: every `.wav` under `jvnv_v1/<speaker>/<emotion>/<session>/` (48 kHz mono 24-bit, 3.19 to 19.78 s); hours from the file lengths. Speakers are `F1`, `F2`, `M1` and `M2`.

### Files per label

| label (emotion in the file name) | samples | hours |
|---|---|---|
| anger | 249 | 0.528 |
| disgust | 258 | 0.590 |
| fear | 265 | 0.648 |
| happy | 280 | 0.746 |
| sad | 257 | 0.822 |
| surprise | 306 | 0.610 |

How these were counted: the emotion field of each file name `<speaker>_<emotion>_<session>_<item>.wav`; the session is `regular` (1,375 files; the nonverbal phrase was designated) or `free` (240; the speaker chose it). The release has no neutral class. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- No byte-identical or near-identical recordings were found. The release holds a transcription table (`transcription.csv`) and labels of the nonverbal sections (`nv_label/`).


## Download

`prepare_source` downloads `jvnv_ver1.zip` (1.8 GB) from the authors' site and
extracts only the WAV files. The corpus is CC BY-SA 4.0. No checksum is
published, so completeness is checked by counting the 1,615 files.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share all
4 speakers.

| split | utterances |
|---|---|
| train | 1130 |
| valid | 162 |
| test | 323 |

## Listening test

The paper reports a crowd-sourced forced-choice test: 60 clips with balanced emotions, 30 clips for each of 30 workers; the accuracy is 94.21% (87.37% when the nonverbal part is left out). The individual results are not distributed.

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
