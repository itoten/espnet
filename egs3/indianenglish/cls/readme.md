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

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Mendeley Data record `mtk28hgc6x` version 1 (`Original_dataset.zip` and `Augmented.zip`, published 2025-06-09; downloaded around 2026-10-15). The counts below are those of `Original_dataset.zip`. All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Mendeley Data, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 747 | unknown | 0.59 |

How these were counted: every `.wav` in `Original_dataset.zip` (the files are Ogg Opus streams with a `.wav` extension: 48 kHz mono, 1.14 to 7.44 s, median 2.74 s); hours from the file lengths. The release does not say who spoke which file, so the number of speakers is unknown. `Augmented.zip` holds 2,241 further recordings, none of which is an original recording (compared by CRC32 and size); they were not counted.

### Files per label

| label (emotion folder in the release) | samples | hours |
|---|---|---|
| Happy | 226 | 0.189 |
| Sad | 176 | 0.150 |
| Neutral | 175 | 0.133 |
| Angry | 170 | 0.122 |

How these were counted: the emotion folder each file is in. The release has no disgust, fear or surprise class. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The description says the original set has 1,000 samples and the augmented set 3,000; the zips hold 747 and 2,241. The `.wav` files are Ogg Opus.
- 33 groups (83 files) are byte-identical; in 2 groups the labels differ (`Angry/046` and `Neutral/002`, `Angry/071` and `Sad/085`).
- The zip's `README.md` says each emotion is spoken from a fixed list of 10 sentences (the neutral list shows 9) by multiple actors; the description says native Indian English speakers. Whether the speech is acted is not stated.


## What is used

Only `Original_dataset.zip` is used. The dataset's other zip, `Augmented.zip`, is
the augmented set the description mentions (pitch shifting, time
stretching and noise addition, 3,000 samples). It holds 2,241 recordings, three times
the original 747, and none of the original recordings (compared by CRC32 and size);
it is not used.

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

**Duplicates.** `exclude_duplicates` in `dataset/config.yaml` (default `true`) keeps one recording per group of byte-identical files with one label (48 extra copies are dropped); `false` keeps every copy. Copies with conflicting labels are always dropped.
