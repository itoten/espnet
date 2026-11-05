# BanglaMood speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BanglaMood alone and scores it on its own test split, classifying each
recording into one of eight emotions.

[BanglaMood: A Rich Audio Dataset for Eight Emotion Recognition in Bangla
Speech](https://data.mendeley.com/datasets/9cc8k5v7w4/3) (Patwary, Hossain, Hossain and
Rahman, North South University; Mendeley Data, V3, doi:10.17632/9cc8k5v7w4.3, published
2025-07-02) is Bangla speech cut from YouTube videos. The dataset description says:

> The BanglaMOOD dataset comprises 3,220 audio files in WAV format, each capturing vocal
> expressions of human emotion in Bangla. The dataset features eight emotion categories:
> Angry, Happy, Sad, Neutral, Surprised, Fear, Disappointment, and Excitement. Each audio
> sample ranges from 1 to 5 seconds in duration, with all files standardized to 5 seconds
> for consistency in feature extraction. The audio was collected from a wide variety of
> real-life contexts, including short stories, vlogs, and sports highlights sourced from
> YouTube, ensuring a broad range of expressive and natural vocalizations. Both male and
> female voices are included to provide gender diversity.

The labels are `angry`, `neutral`, `fear`, `disappointed`, `sad`, `surprise`, `happy` and
`excited`; there is no `disgust`. `disappointed` and `excited` are not among the seven common
labels of the other recipes; they are kept as classes of their own (the folders are named
`Disappointment` and `Excitement`, and `Surprised` becomes `surprise`). No journal article
describes the dataset (the Mendeley record lists none), so the record is the only source.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Mendeley Data record `9cc8k5v7w4` version 3 (BanglaMood, published 2025-07-02; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Mendeley Data, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 3,220 | unknown | 2.90 |

How these were counted: every `.wav` in `<Emotion>/<Gender>/` (48 kHz mono 16-bit, 0.52 to 7.57 s, mean 3.24 s; 2,190 under `Male` and 1,030 under `Female`); hours from the file lengths. The release does not say who spoke which clip, so the number of speakers is unknown.

### Files per label

| label (emotion folder in the release) | samples | hours |
|---|---|---|
| Angry | 611 | 0.504 |
| Neutral | 610 | 0.574 |
| Fear | 449 | 0.416 |
| Disappointment | 343 | 0.302 |
| Sad | 337 | 0.321 |
| Surprised | 303 | 0.244 |
| Happy | 300 | 0.283 |
| Excitement | 267 | 0.257 |

How these were counted: the emotion folder each file is in; the counts equal those in the description. The release has no disgust class. The description says three native Bangla annotators labelled the clips from tone, expression, situation, screams, cries and meaning, and that the videos were divided among them; whether more than one annotator judged a clip is not stated. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The description says all clips were standardized to 5 seconds; the files are 0.52 to 7.57 s long.
- Two files have identical content (SHA-256) but different labels: `Disappointment/Male/0737.wav` and `Neutral/Male/0704.wav`.
- The clips are cut from public YouTube videos (short stories, vlogs and sports highlights); no source videos or their rights are listed, and no journal article describes the dataset.


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

## Listening test

The labels were given by people, and they could see the video. The record's "method" field
says:

> We collected our data from YouTube by randomly choosing videos such as short stories or
> vlogs from publicly available videos. Multiple videos were distributed to 3 native
> Bangla-speaking annotators who were given detailed instructions to be consistent in
> labelling all the clips for higher accuracy and quality. [...] They extracted audio voices
> between 1 to 5 seconds in length and labelled the audio file according to the emotion they
> deduced upon considering the tone, facial expressions, scenario, screams, cries, and
> meanings of the words spoken.

So the videos were divided among three annotators; the text does not say that each clip was
judged by more than one of them, and it gives no agreement coefficient or accuracy. No ratings
are distributed. Because the annotators also used facial expressions and the scenario, a label
may not be recoverable from the audio alone.

## Reference

Patwary, M. S. A., Hossain, M., Hossain, M. M. and Rahman, R. M. (2025). *BanglaMood: A Rich
Audio Dataset for Eight Emotion Recognition in Bangla Speech*. Mendeley Data, V3.
doi:10.17632/9cc8k5v7w4.3.

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
