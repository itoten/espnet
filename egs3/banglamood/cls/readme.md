# BanglaMood speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BanglaMood alone and scores it on its own test split, classifying each
recording into one of eight emotions.

[BanglaMood: A Rich Audio Dataset for Eight Emotion Recognition in Bangla
Speech](https://data.mendeley.com/datasets/9cc8k5v7w4/3) (Patwary, Hossain, Hossain
and Rahman, North South University; Mendeley Data, V3, doi:10.17632/9cc8k5v7w4.3)
is Bangla speech cut from YouTube videos (short stories, vlogs and sports
highlights). According to the dataset description, three native Bangla speakers
labeled each clip by considering the tone, facial expressions, scenario, screams,
cries and meaning of the words, so the labels do not rest on the audio alone. No
journal article for the dataset was found. The labels are `angry`, `neutral`, `fear`,
`disappointed`, `sad`, `surprise`, `happy` and `excited`. There is no `disgust`.
`disappointed` and `excited` are not among the seven common labels of the other
recipes; they are kept as classes of their own (the folders are named
`Disappointment` and `Excitement`, and `Surprised` becomes `surprise`).

## What is used

The release has 3,220 WAV files in `<Emotion>/<Gender>/<NNNN>.wav` (for example
`Angry/Female/0160.wav`). The file numbers are unique across the release, and the
per-folder counts equal those in the description. All files are 48 kHz mono 16-bit
PCM. The description says all clips are standardized to 5 seconds; the files are
0.52 to 7.57 seconds long (mean 3.24, median 3.24). The gender folder is not used
as a label.

**Two files have identical content (SHA-256) but different labels**:
`Disappointment/Male/0737.wav` and `Neutral/Male/0704.wav`. Neither is used, so
3,218 recordings (2.90 hours) remain.

`create_dataset` converts the clips to 16 kHz mono 16-bit WAV with ffmpeg;
`remove_long_short` drops nothing.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BANGLAMOOD` — the directory holding the `Angry/`, `Happy/`, ... folders, or where to download them to
- `BANGLAMOOD_OUTPUT` — where the converted audio and manifests are written

## Download

The Mendeley Data file listing gives each recording's size and SHA-256 but not its
folder, and the folder is the label. So `prepare_source` downloads the dataset's
zip (898 MB, the "Download all" file of the dataset page), checks every recording in
it against the SHA-256 digests of the listing (each digest used exactly as often as
listed), and writes the recordings into `<Emotion>/<Gender>/`. Mendeley rejects
requests that do not send a browser-like `User-Agent`, so the recipe sends one. The
dataset is licensed CC BY 4.0 (the license field of the Mendeley listing). Please
cite the Mendeley record above. The clips come from third-party videos; whether
their rights were cleared is not stated in the record.

## Splits

The release publishes no partition and does not identify speakers. Recordings are
split 70/10/20 at random within each label (`train_ratio`, `valid_ratio`,
`test_ratio`, `split_seed` in `dataset/config.yaml`), so the splits probably share
speakers and source videos (the release does not say which video a clip came from).

| split | recordings | after `remove_long_short` | angry | disappointed | excited | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 2253 | 2253 | 428 | 240 | 187 | 314 | 210 | 426 | 236 | 212 |
| valid | 322 | 322 | 61 | 34 | 27 | 45 | 30 | 61 | 34 | 30 |
| test | 643 | (not filtered) | 122 | 68 | 53 | 90 | 60 | 122 | 67 | 61 |

`angry` and `neutral` are about twice as large as `excited`, so the class weights
(written by `prepare_labels`) range from 0.658 (angry) to 1.51 (excited). Report
Macro-F1 rather than accuracy.

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of the 898 MB zip, SHA-256 checks, extraction of 3,220 files) | 292.3 s |
| `create_dataset` (conversion to 16 kHz mono, one file at a time) | 143.9 s |
| `remove_long_short` | 14.3 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 2 min 44.9 s (includes starting `pixi` and Python) |

The downloaded audio takes 963 MB and the converted data 327 MB.

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
