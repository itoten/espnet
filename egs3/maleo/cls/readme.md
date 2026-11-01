# Maleo Emotion speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Maleo Emotion alone and scores it on its own test split, classifying each
recording into one of seven emotions.

Maleo Emotion (`maleo-ai/maleo-emotion` on Hugging Face, CC BY 4.0) is Indonesian
speech: 700 clips of about 3 seconds, 100 for each of `angry`, `disgust`, `fear`,
`happy`, `neutral`, `sad` and `surprise`, the same seven labels the other recipes
share. According to the dataset card of the copy used here, the clips are cut
from YouTube videos of spontaneous speech, the label is the dominant emotion
perceived by listening (the card does not say who listened or how many people; no
agreement figure is given; some clips mix two emotions), and some clips are code-mixed or entirely in a foreign
language. The speakers are not identified.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `dikal/maleo-emotion-annotated` (revision `ddbaf824d3632b3fcaf1326b7ca9ed4c2b0e0114`; downloaded around 2026-10-15). This is a copy by another account of `maleo-ai/maleo-emotion`, which requires a login; the 700 recordings were checked to be identical in path and content, and the copy adds transcripts and word timestamps. All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face (the copy), direct download, no sign-in. The original repository `maleo-ai/maleo-emotion` requires a login.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 700 | unknown | 0.59 |

How these were counted: every audio file under `audio/<emotion>/` (44.1 kHz stereo as stated; 1.50 to 4.46 s, median 3.0 s); hours from the file lengths. The release gives no speaker information.

### Files per label

| label (folder name in the release) | read as (Indonesian) | samples |
|---|---|---|
| Jijik | disgust | 100 |
| Marah | angry | 100 |
| Netral | neutral | 100 |
| Sedih | sad | 100 |
| Senang | happy | 100 |
| Takut | fear | 100 |
| Terkejut | surprise | 100 |

How these were counted: the folder each file is in. The card says the label is the dominant emotion perceived by listening; it does not say who listened or how many people, and gives no agreement figure; some clips mix two emotions.

### Issues in the release

- The card says 16-bit PCM, 44.1 kHz stereo; the files are mixed: the 40 `neutral` clips are compressed audio (20 MP3 and 20 AAC, with a `.WAV` extension) and no other label has any; 20 `happy` clips are mono and 16 are 48 kHz.
- Four clips marked `[NO_SPEECH]` in the transcripts (disgust 2, fear 1, sad 1) contain no human voice but carry an emotion label; 27 clips marked `[INAUDIBLE]` contain vocalizations such as crying or shouting.
- The clips are cut from YouTube videos in natural conditions; the card names no source videos and says nothing about their rights, and there is no paper. The speech is colloquial Indonesian, often code-mixed; some clips are entirely in a foreign language.


## What is used

The copy [`dikal/maleo-emotion-annotated`](https://huggingface.co/datasets/dikal/maleo-emotion-annotated),
pinned to one commit (`revision` in `dataset/config.yaml`). It adds transcripts
and word timestamps to the original, which this recipe does not use. All 700
recordings (0.59 hours) are used. The label is the name of the folder the file is
in (`Jijik`, `Marah`, `Netral`, `Sedih`, `Senang`, `Takut`, `Terkejut`, which are
Indonesian for disgust, angry, neutral, sad, happy, fear and surprise). The files
are 44.1 kHz stereo; `create_dataset` converts them to 16 kHz mono 16-bit WAV.
Durations range from 1.5 to 4.5 seconds (median 3.0), so `remove_long_short`
drops nothing.

**Formats and labels.** The card says 16-bit PCM, 44.1 kHz stereo, but the files are
mixed: the 40 `neutral` clips are compressed audio (20 MP3 and 20 AAC, with a `.WAV`
extension) and no other label has any, and 20 `happy` clips are mono and 16 are
48 kHz. The format therefore differs by label; all clips are used as they are.
Four clips marked `[NO_SPEECH]` in the copy's transcripts (disgust 2, fear 1, sad 1)
have no human voice but a label; they are used too.

**Source rights.** The clips are cut from YouTube videos. The card of the copy says
only that the data come from YouTube videos in natural conditions and gives CC BY 4.0
as the license; it names no source videos and says nothing about the rights of the
videos, and there is no paper. Whether the clips can be used as open data, and whether
non-commercial terms apply, is unconfirmed.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MALEO` — the directory holding `audio/<emotion>/`, or where to download it to
- `MALEO_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` lists the 700 files of the pinned revision through the Hugging
Face API and downloads them with 8 threads. Hugging Face lists each file's size
and SHA-256, and both are checked. No login is needed for this copy; the original
repository asks for one. Because the copy is not the original authors' upload,
the copy's author could have changed the files. The Hugging Face LFS hashes of all
700 audio files are the same in the copy and in the original, so the audio is the same.

## Splits

The release publishes no partition. Recordings are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`), which gives every label 70, 10 and 20 recordings.
Speakers are unknown and cannot be held out, so the splits probably share
speakers.

| split | recordings | after `remove_long_short` |
|---|---|---|
| train | 490 | 490 |
| valid | 70 | 70 |
| test | 140 | (not filtered) |

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network, and a 28 GB download was running at the same time.

| step | time |
|---|---|
| `prepare_source` (700 files, 337 MB, 8 threads, SHA-256 checked) | 2 min 20 s |
| `create_dataset` (conversion to 16 kHz, one file at a time) | 43.4 s |
| `remove_long_short` | 16.4 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 1 min 8 s (includes starting `pixi` and Python) |

The downloaded audio takes 337 MB and the converted data 67 MB.

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
