# Indian English speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on the Indian English Speech Emotion Dataset alone and scores it on its own
test split, classifying each recording into one of four emotions.

[Indian English Speech Emotion Dataset](https://data.mendeley.com/datasets/mtk28hgc6x/1)
(Mendeley Data, V1, doi:10.17632/mtk28hgc6x.1, published 2025-06-09) holds
English speech with Indian accents labeled `angry`, `happy`, `neutral` or `sad`.
There is no `disgust`, `fear` or `surprise`. The contributors listed on Mendeley
Data are harsh harsh, Yash Singh, Nikhil Singh and Ayush Bhardwaj (BMS College of
Engineering). No paper is linked to the dataset.

## What is used

Only `Original_dataset.zip` is used. The dataset's other zip, `Augmented.zip`, is
presumably the augmented set the description mentions (pitch shifting, time
stretching and noise addition, 3,000 samples); its content was not inspected and it is not used.

The zip holds 747 recordings, in one folder per emotion (`Happy/001.wav`, ...). The
dataset description says the original set has 1,000 samples; the zip does not, and
the difference was not investigated. Two properties of the files matter:

- **The `.wav` files are Ogg Opus streams** (checked with `ffprobe`: container
  `ogg`, codec `opus`, 48 kHz mono). ffmpeg reads them; `create_dataset` converts
  them to 16 kHz mono WAV.
- **Some files repeat.** 50 of the 747 have the same content (SHA-256) as another
  file. Where the copies share a label, the first by folder and name is kept. Two
  pairs have the same content under *different* labels (`Angry/046.wav` and
  `Neutral/002.wav`; `Angry/071.wav` and `Sad/085.wav`); the label is then
  unknown, so neither copy of those pairs is used. This leaves 695 recordings
  (0.548 hours).

Durations range from 1.14 to 7.44 seconds (median 2.74), so `remove_long_short`
drops nothing.

The zip's `README.md` says every emotion is spoken from a fixed list of 10
sentences (the neutral list shows 9) "by multiple actors". So the same sentences
occur in train, valid and test, and the dataset description's wording ("native
Indian English speakers") and the README's ("actors") do not settle whether the
speech is acted. The release does not say who spoke which file.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `INDIANENGLISH` — the directory holding the `Angry/`, `Happy/`, `Neutral/` and `Sad/` folders, or where to download them to
- `INDIANENGLISH_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` reads the Mendeley Data API listing, downloads `Original_dataset.zip`
(4.97 MB), checks its size and SHA-256 against the listing, extracts only the
`<Emotion>/<digits>.wav` members, and deletes the zip. Mendeley rejects requests
that do not send a browser-like `User-Agent`, so the recipe sends one. The dataset
is licensed CC BY 4.0 (the license field of the Mendeley listing). Please cite the
dataset as above.

## Splits

The release publishes no partition and does not identify speakers. Recordings are
split 70/10/20 at random within each label (`train_ratio`, `valid_ratio`,
`test_ratio`, `split_seed` in `dataset/config.yaml`), so the splits probably share
speakers and certainly share sentences.

| split | recordings | after `remove_long_short` | angry | happy | neutral | sad |
|---|---|---|---|---|---|---|
| train | 486 | 486 | 117 | 131 | 121 | 117 |
| valid | 70 | 70 | 17 | 19 | 17 | 17 |
| test | 139 | (not filtered) | 33 | 37 | 35 | 34 |

The classes are nearly balanced (class weights 0.93 to 1.04). The test set is small
(33 to 37 per class), so scores vary a lot between runs.

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`).

| step | time |
|---|---|
| `prepare_source` (4.97 MB download, SHA-256 check, extraction of 747 files) | 6.2 s |
| `create_dataset` (conversion to 16 kHz, one file at a time) | 32.1 s |
| `remove_long_short` | 17.9 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 58.2 s (includes starting `pixi` and Python) |

The extracted audio takes 6.2 MB and the converted data 62 MB.

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
