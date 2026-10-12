# INESCO speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on INESCO alone and scores it on its own test split, classifying each
recording into one of three emotions.

[INESCO: Indonesian Expressive Speech Corpus for Emotion Speech
Synthesis](https://data.mendeley.com/datasets/hzrznx3xs5/1) (Mendeley Data, V1) is
an Indonesian corpus built for speech synthesis. According to the dataset
description, four professional theater artists (two women, two men; members of the
Indonesian Arts Council) read 600 phonetically balanced sentences in a soundproof
studio, each sentence in one emotion. The description reports 3 hours 1 minute 49
seconds in total. No journal article for the dataset was found. The labels are
`happy`, `angry` and `sad`; there is no `neutral`, `fear`, `disgust` or `surprise`.

## What is used

The release has 2,399 WAV files named `<speaker>_<emotion><sentence>.wav` (for
example `mdpa_a250.wav`): speaker `mdpa` and `mbaz` (men) or `fyat` and `fcim`
(women), emotion letter `h` (happiness), `a` (anger) or `s` (sadness), then the
sentence number 001 to 600. The release's `Guideline of File Naming INESCO
Dataset.pdf` assigns sentence numbers 001-211 to happiness, 212-401 to anger and
402-600 to sadness, and all 2,399 file names agree with that. So each sentence is
read in a single emotion, and the three emotions use different sentences. A
classifier can therefore also learn which sentences belong to which emotion; the
splits below keep every sentence in one split to avoid rewarding that.

One recording is missing (`fcim_a332.wav`, so 599 files for `fcim`), and the
description's count of 2,400 does not match. One more, `mbaz_h138.wav`, has lost the
first 8 bytes of its header (it begins with `WAVE` instead of `RIFF`) and ffmpeg
cannot open it, so it is skipped. 2,398 recordings (3.02 hours, 1.1 to 11.8 seconds,
median 4.37) remain. The source files are 44.1 kHz 16-bit PCM; the recordings of
`fyat` and `mdpa` are mono, and those of `fcim` and `mbaz` are stereo (the channels
differ by at most 5 in 16-bit units in the 200 stereo files compared). No two files
have the same content (SHA-256). `create_dataset` converts them to 16 kHz mono 16-bit
WAV with ffmpeg; `remove_long_short` drops nothing.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `INESCO` — the directory holding `audio/*.wav`, or where to download it to
- `INESCO_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the dataset's zip from Mendeley Data (1.4 GB, the
"Download all" file of the dataset page), reads each recording's size and SHA-256
from the Mendeley Data API listing, and writes only recordings that match it.
Mendeley rejects requests that do not send a browser-like `User-Agent`, so the
recipe sends one. The dataset is licensed CC BY 4.0 (the license field of the
Mendeley listing). Please cite the Mendeley record above.

## Splits

The release publishes no partition. Because all four speakers read each sentence in
the same emotion, the 600 sentence numbers (not the recordings) are split 70/10/20 at
random within each emotion (`train_ratio`, `valid_ratio`, `test_ratio`,
`split_seed` in `dataset/config.yaml`), so no sentence appears in two splits. The
speakers are not held out: all four are in every split.

| split | recordings | after `remove_long_short` | angry | happy | sad |
|---|---|---|---|---|---|
| train | 1678 | 1678 | 531 | 591 | 556 |
| valid | 240 | 240 | 76 | 84 | 80 |
| test | 480 | (not filtered) | 152 | 168 | 160 |

The classes are nearly balanced (class weights 0.946 to 1.053, written by
`prepare_labels`). Report Macro-F1 rather than accuracy.

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of the 1.4 GB zip, SHA-256 checks, extraction of 2,399 files) | 291.1 s |
| `create_dataset` (conversion to 16 kHz mono, one file at a time) | 122.0 s |
| `remove_long_short` | 14.4 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 2 min 22.9 s (includes starting `pixi` and Python) |

The downloaded audio takes 1.4 GB and the converted data 339 MB.

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
