# IndoWaveSentiment speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on IndoWaveSentiment alone and scores it on its own test split, classifying
each recording into one of five emotions.

[IndoWaveSentiment](https://data.mendeley.com/datasets/j9ytfdzy27/1) (Mendeley Data,
V1, doi:10.17632/j9ytfdzy27.1, published 2024-06-24) is Indonesian speech from 10
actors, five men and five women. The contributors listed on Mendeley Data are
Anugrayani Bustamin, Andi M Rizky, Elly Warni, Intan Sari Areni and Indrabayu
Indrabayu. The labels are `neutral`, `happy`, `surprise`, `disgust` and
`disappointed`. There is no `angry`, `fear` or `sad`, and `disappointed` has no
counterpart in the other recipes.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Mendeley Data record `j9ytfdzy27` version 1 (IndoWaveSentiment, published 2024-06-24; `audio/` and the naming-guidelines PDF; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Mendeley Data, no sign-in (the listing gives each file's size and SHA-256, which were checked).

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 300 | 10 | 0.26 |

How these were counted: every `.wav` in `audio/` (mono 16-bit; 44.1 kHz in 210 files and 48 kHz in 90; 3.0 to 3.8 s, of which 236 are exactly 3.0 s); hours from the file lengths. Speakers are actors 01 to 10 of the file names `<actor>-<emotion>-<intensity>-<repetition>.wav` (odd numbers men, even numbers women, per the guidelines PDF).

### Files per label

| emotion code in the file name | read as (naming guidelines) | samples | hours |
|---|---|---|---|
| 01 | neutral | 60 | 0.050 |
| 02 | happy | 60 | 0.050 |
| 03 | surprise | 60 | 0.050 |
| 04 | disgust | 60 | 0.055 |
| 05 | disappointed | 60 | 0.054 |

How these were counted: the second field of each file name (10 actors x 5 emotions x 2 intensities (01 normal, 02 strong) x 3 repetitions). The release has no angry, fear or sad class. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The description says the audio was recorded at 16 kHz with 32-bit depth; the files are 44.1 or 48 kHz, mono, 16-bit PCM.
- No two files have the same content (SHA-256).

### Not stated in the documentation

- A Whisper transcription of 12 randomly chosen recordings gave the same sentence for all of them, "Kualitas HP ini cukup bagus." (Indonesian: "The quality of this phone is quite good"); automatic transcription only, not checked by a person. The description says the same sentence is repeated but does not give it.


## What is used

All 300 recordings: 10 actors × 5 emotions × 2 intensities × 3 repetitions. The
file names follow the release's `IndoWaveSentiment File Naming Guidelines.pdf`:
`<actor>-<emotion>-<intensity>-<repetition>.wav` (for example `07-03-02-03.wav`),
with actors 01-10 (odd numbers men, even numbers women), emotion codes 01 neutral,
02 happy, 03 surprise, 04 disgust, 05 disappointed, and intensity 01 normal or 02
strong. The label is the emotion code; the intensity is not used. No two files have
the same content (SHA-256).

The dataset description says the audio was recorded at 16 kHz with 32-bit depth.
The files do not match: 210 are 44.1 kHz and 90 are 48 kHz, all mono 16-bit PCM
(checked with `soundfile`). `create_dataset` converts them to 16 kHz mono 16-bit
WAV with ffmpeg. Durations after conversion range from 3.00 to 3.80 seconds (median
3.00), so `remove_long_short` drops nothing. The description says each actor
repeated "the same sentence" three times across the five emotions; the sentence
is not given. A Whisper transcription of 12 randomly chosen recordings gave the same sentence for all of them, "Kualitas HP ini cukup bagus." (Indonesian, "The quality of this phone is quite good"); this comes from the automatic transcription only and was not checked by a person.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `INDOWAVESENTIMENT` — the directory holding `audio/`, or where to download it to
- `INDOWAVESENTIMENT_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` reads the Mendeley Data API listing, downloads the 300 WAV files
(82 MB) with 8 threads, and checks each one's size and SHA-256 against the listing.
Mendeley rejects requests that do not send a browser-like `User-Agent`, so the
recipe sends one. The dataset is licensed CC BY 4.0 (the license field of the
Mendeley listing). Please cite the dataset as above.

## Splits

The release publishes no partition. Recordings are split 70/10/20 at random within
each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). The actors are not held out, so every actor appears in
train, valid and test.

| split | recordings | after `remove_long_short` | neutral | happy | surprise | disgust | disappointed |
|---|---|---|---|---|---|---|---|
| train | 210 | 210 | 42 | 42 | 42 | 42 | 42 |
| valid | 30 | 30 | 6 | 6 | 6 | 6 | 6 |
| test | 60 | (not filtered) | 12 | 12 | 12 | 12 | 12 |

The classes are exactly balanced (class weights all 1.0). The test set is tiny (12
per class), so scores vary a lot between runs. Because the set is small,
`max_epochs` in `conf/training.yaml` is 429 (the rule that gives about 3000
optimizer steps; see the comment there).

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download ran
while two other Mendeley downloads were running, so it depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of 300 files, 82 MB, with size and SHA-256 checks) | 85.6 s |
| `create_dataset` (conversion to 16 kHz, one file at a time) | 16.2 s |
| `remove_long_short` | 16.8 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 40.5 s (includes starting `pixi` and Python) |

The downloaded audio takes 82 MB and the converted data 30 MB.

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
